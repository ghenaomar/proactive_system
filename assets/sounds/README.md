# Alert Sounds Directory

This directory contains audio files for the Alert Manager system.

## Required Sound Files

Place the following sound files in this directory:

```
assets/sounds/
├── alert_info.mp3         # Informational alert (very low severity)
├── alert_low.mp3          # Low severity alert
├── alert_medium.mp3       # Medium severity alert
├── alert_high.mp3         # High severity alert
├── alert_critical.mp3     # Critical alert (requires immediate attention)
└── README.md             # This file
```

## Sound Specifications

### Recommended Specifications
- **Format**: MP3 or WAV
- **Duration**: 0.5 - 2.0 seconds (short and distinct)
- **Sample Rate**: 44.1 kHz or 48 kHz
- **Bit Rate**: 128 kbps or higher
- **Volume**: Normalized to avoid clipping

### Sound Characteristics by Severity

| Severity | Description | Suggested Sound |
|----------|-------------|-----------------|
| `info` | Informational notification | Soft beep or chime |
| `low` | Minor attention needed | Single tone, calm |
| `medium` | Moderate concern | Double beep, noticeable |
| `high` | Serious issue | Triple beep, urgent |
| `critical` | Immediate action required | Alarm sound, very urgent |

## Fallback Behavior

If sound files are not provided:
- **Windows**: System will use `winsound.Beep()` with varying frequencies
- **Linux/Mac**: System will use `pygame.mixer` with generated tones
- **No audio hardware**: Alerts will be logged but no sound will play

## How to Generate Sounds

### Option 1: Free Sound Resources
- [Freesound.org](https://freesound.org) - Search for "alert", "beep", "notification"
- [ZapSplat](https://www.zapsplat.com) - Free sound effects
- [Mixkit](https://mixkit.co/free-sound-effects/) - Free sound effects library

### Option 2: Generate with Python
```python
# Example: Generate simple beeps with different frequencies
import numpy as np
from scipy.io.wavfile import write

def generate_beep(filename, frequency, duration=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * frequency * t)
    wave = (wave * 32767).astype(np.int16)
    write(filename, sample_rate, wave)

# Generate alerts
generate_beep("alert_low.wav", 440, 0.3)      # A4 note
generate_beep("alert_medium.wav", 523, 0.4)   # C5 note
generate_beep("alert_high.wav", 659, 0.5)     # E5 note
generate_beep("alert_critical.wav", 880, 0.7) # A5 note
```

### Option 3: Use Text-to-Speech (for development)
```python
# Using gTTS (Google Text-to-Speech)
from gtts import gTTS

alerts = {
    "info": "Information",
    "low": "Low priority alert",
    "medium": "Medium priority alert",
    "high": "High priority alert",
    "critical": "Critical alert"
}

for level, text in alerts.items():
    tts = gTTS(text=text, lang='en')
    tts.save(f"alert_{level}.mp3")
```

## Testing Sounds

Test the alert system with:

```bash
# Test compute_decisions with sound enabled
python tools/compute_decisions.py \
  --run_dir outputs/runs/YOUR_RUN \
  --sound 1 \
  --sounds_dir assets/sounds
```

## Customization

You can customize alert sounds by:
1. Placing your own sound files in this directory
2. Using the naming convention above
3. The Alert Manager will automatically use them

## License

Make sure any sound files you use comply with licensing requirements:
- Use royalty-free sounds
- Check Creative Commons licenses
- Provide attribution if required

## Notes

- Keep sounds short (< 2 seconds) to avoid disrupting the monitoring flow
- Test volume levels to ensure they're audible but not startling
- Consider the exam environment - sounds should be attention-grabbing but professional
- You can disable sounds by passing `--sound 0` to `compute_decisions.py`
