from __future__ import annotations

from proctor_ai.events.presence import PresenceEventTracker


def test_presence_emits_only_on_change_enter_and_exit() -> None:
    tr = PresenceEventTracker(student_ids=["s1"])

    # frame 0: absent -> present => enter
    ev = tr.update({"s1": True}, frame_index=0, ts_ms=0)
    assert len(ev) == 1
    assert ev[0].student_id == "s1"
    assert ev[0].event_type == "enter"
    assert ev[0].present is True

    # frame 1: still present => no event
    ev = tr.update({"s1": True}, frame_index=1, ts_ms=33)
    assert len(ev) == 0

    # frame 2: present -> absent => exit
    ev = tr.update({"s1": False}, frame_index=2, ts_ms=66)
    assert len(ev) == 1
    assert ev[0].event_type == "exit"
    assert ev[0].present is False

    c = tr.counts()
    assert int(c["s1"]) == 2

    s = tr.state()
    assert s["s1"] is False
