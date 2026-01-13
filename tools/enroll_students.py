#!/usr/bin/env python3
"""Student Enrollment CLI Tool.

This tool provides commands for enrolling students into the identity
database and managing their reference photos.

Usage:
    # Enroll a single student
    python tools/enroll_students.py enroll \
        --student-id "12345" \
        --name "Ahmed Mohamed" \
        --zone "seat1" \
        --photos photo1.jpg photo2.jpg

    # Bulk enroll from directory
    python tools/enroll_students.py bulk --dir prepared_students/

    # Compute embeddings for all students
    python tools/enroll_students.py compute-embeddings

    # List all enrolled students
    python tools/enroll_students.py list

    # Verify a face against the database
    python tools/enroll_students.py verify --image face.jpg

Author: Proctor AI Team
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_database_path(args: argparse.Namespace) -> Path:
    """Get database path from args or default."""
    if hasattr(args, "db_path") and args.db_path:
        return Path(args.db_path)
    return project_root / "data" / "students"


def cmd_enroll(args: argparse.Namespace) -> int:
    """Enroll a single student."""
    from proctor_ai.identity import (
        FaceEmbedder,
        StudentDatabase,
        StudentEnroller,
    )
    
    db_path = get_database_path(args)
    db_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing face embedder...")
    embedder = FaceEmbedder(device=args.device)
    
    database = StudentDatabase(db_path)
    database.load()
    
    enroller = StudentEnroller(embedder, database)
    
    try:
        student = enroller.enroll(
            student_id=args.student_id,
            name=args.name,
            zone_id=args.zone,
            photos=args.photos,
            validate=not args.no_validate,
        )
        logger.info(f"✅ Successfully enrolled: {student.name}")
        logger.info(f"   Folder: {student.folder_name}")
        logger.info(f"   Zone: {student.zone_id}")
        logger.info(f"   Photos: {len(student.photos)}")
        return 0
    except Exception as e:
        logger.error(f"❌ Enrollment failed: {e}")
        return 1


def cmd_bulk(args: argparse.Namespace) -> int:
    """Bulk enroll from directory."""
    from proctor_ai.identity import (
        FaceEmbedder,
        StudentDatabase,
        enroll_from_directory,
    )
    
    source_dir = Path(args.dir)
    if not source_dir.exists():
        logger.error(f"Directory not found: {source_dir}")
        return 1
    
    db_path = get_database_path(args)
    db_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing face embedder...")
    embedder = FaceEmbedder(device=args.device)
    
    database = StudentDatabase(db_path)
    database.load()
    
    enrolled = enroll_from_directory(source_dir, embedder, database)
    
    logger.info(f"✅ Enrolled {len(enrolled)} students")
    for student in enrolled:
        logger.info(f"   - {student.name} ({student.zone_id})")
    
    return 0


def cmd_compute_embeddings(args: argparse.Namespace) -> int:
    """Compute embeddings for all students."""
    from proctor_ai.identity import (
        FaceEmbedder,
        StudentDatabase,
        compute_all_embeddings,
    )
    
    db_path = get_database_path(args)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1
    
    logger.info("Initializing face embedder...")
    embedder = FaceEmbedder(device=args.device)
    
    database = StudentDatabase(db_path)
    count = database.load()
    logger.info(f"Loaded {count} students")
    
    processed = compute_all_embeddings(database, embedder)
    logger.info(f"✅ Computed embeddings for {processed} students")
    
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all enrolled students."""
    from proctor_ai.identity import StudentDatabase
    
    db_path = get_database_path(args)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1
    
    database = StudentDatabase(db_path)
    count = database.load()
    
    if count == 0:
        logger.info("No students enrolled")
        return 0
    
    print(f"\n{'ID':<15} {'Name':<25} {'Zone':<10} {'Photos':<8}")
    print("-" * 60)
    
    for student in database.list_all():
        has_emb = "✓" if student.embeddings is not None else "✗"
        print(
            f"{student.student_id:<15} "
            f"{student.name:<25} "
            f"{student.zone_id:<10} "
            f"{len(student.photos):<8} {has_emb}"
        )
    
    print(f"\nTotal: {count} students")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a face against the database."""
    from proctor_ai.identity import (
        FaceEmbedder,
        StudentDatabase,
    )
    import cv2
    
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1
    
    db_path = get_database_path(args)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1
    
    logger.info("Initializing face embedder...")
    embedder = FaceEmbedder(device=args.device)
    
    database = StudentDatabase(db_path)
    database.load()
    
    # Extract face from image
    logger.info(f"Processing: {image_path}")
    faces = embedder.extract_from_file(image_path)
    
    if not faces:
        logger.warning("No face detected in image")
        return 1
    
    face = faces[0]
    logger.info(f"Face detected (confidence: {face.confidence:.2f})")
    
    # Search database
    matches = database.search_by_embedding(
        face.embedding,
        threshold=args.threshold,
        top_k=3,
    )
    
    if not matches:
        logger.warning("❌ No match found - UNKNOWN PERSON")
        return 0
    
    print(f"\n{'Match':<5} {'Name':<25} {'ID':<15} {'Similarity':<12}")
    print("-" * 60)
    
    for i, (student, sim) in enumerate(matches, 1):
        status = "✅" if sim >= 0.5 else "⚠️"
        print(f"{status} {i:<3} {student.name:<25} {student.student_id:<15} {sim:.3f}")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Student Enrollment CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        help="Path to student database (default: data/students)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Computation device",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a single student")
    enroll_parser.add_argument("--student-id", required=True, help="Student ID")
    enroll_parser.add_argument("--name", required=True, help="Student name")
    enroll_parser.add_argument("--zone", required=True, help="Zone/seat ID")
    enroll_parser.add_argument("--photos", nargs="+", required=True, help="Photo paths")
    enroll_parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    
    # Bulk command
    bulk_parser = subparsers.add_parser("bulk", help="Bulk enroll from directory")
    bulk_parser.add_argument("--dir", required=True, help="Source directory")
    
    # Compute embeddings command
    compute_parser = subparsers.add_parser(
        "compute-embeddings",
        help="Compute embeddings for all students",
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List enrolled students")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a face")
    verify_parser.add_argument("--image", required=True, help="Image path")
    verify_parser.add_argument("--threshold", type=float, default=0.35, help="Match threshold")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "enroll": cmd_enroll,
        "bulk": cmd_bulk,
        "compute-embeddings": cmd_compute_embeddings,
        "list": cmd_list,
        "verify": cmd_verify,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
