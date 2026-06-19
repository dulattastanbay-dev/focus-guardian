# Focus Guardian 👀

A small webcam tool that keeps you honest while you study or work. It watches
for your face at your desk, and if you wander off (or zone out and look away)
for longer than a grace period, it nudges you to get back to it. When you stop,
it prints a focus report and logs the session so you can track your habits over
time.

Everything runs **100% locally** — the camera frames never leave your machine,
nothing is uploaded, and nothing is stored except a small text log of your
session stats.

## Demo

```
[##########----------] focused  focus 0:18:42  nudges 2

  [nudge #2] You've been away for 0:00:23 -- back to work!
```

End-of-session report:

```
==========================================
  SESSION REPORT
==========================================
  Total time      0:45:10
  Focused         0:39:55
  Away            0:05:15
  Focus ratio     88%
  Longest streak  0:12:30
  Nudges          3
==========================================
  Saved to focus_log.json  (7 sessions logged)
```

## How it works

1. Grabs frames from your webcam with **OpenCV**.
2. Tracks your face **and your eyes** with **MediaPipe FaceLandmarker**. If
   MediaPipe isn't installed it falls back to the **YuNet DNN detector**, and
   below that to bundled **Haar cascades** — so it always runs.
3. Smooths detection over time, so a blink or a brief glance doesn't flip the
   state — the big reason tracking feels steady.
4. Distinguishes three states: **focused** (eyes on the screen), **eyes off
   screen** / distracted (you're there but looking away or down at a phone),
   and **away** (you left the desk).
5. If you're away or your eyes leave the screen past the grace period, it
   beeps, shows a **desktop notification**, and prints a nudge.
6. On exit it builds a report and appends it to `focus_log.json`.

## Eye-gaze tracking

With MediaPipe installed, Focus Guardian reads per-eye look-direction
("blendshapes") to know whether your eyes are actually **on the screen** —
not just whether your head is pointed forward. Because the natural gaze angle
depends on where your webcam sits, it **calibrates** at startup:

```
Calibrating -- look straight at your screen for a few seconds.
```

Look normally at your screen during the countdown and it records that as your
baseline; eyes deviating beyond a margin then count as *off screen*. Press
**`k`** anytime to recalibrate (e.g. if you move). Tune strictness with
`--gaze-threshold` (lower = stricter) or turn it off with `--no-gaze`.

> Install MediaPipe with `pip install mediapipe`. Without it, the app still
> runs and falls back to head-pose tracking automatically.

## Pomodoro mode

Run timed work/break cycles. Phase changes beep and pop a notification.

```bash
python focus_guardian.py --pomodoro            # 25 min work / 5 min break
python focus_guardian.py --work 50 --break 10  # custom lengths
```

## Stats dashboard

See your focus history (per-day focus ratio, totals, nudges, pomodoros):

```bash
python focus_guardian.py --stats
```

## Setup

```bash
git clone https://github.com/dulattastanbay-dev/focus-guardian.git
cd focus-guardian
pip install -r requirements.txt
python focus_guardian.py
```

> Requires Python 3.9+ and a webcam.

## Usage

```bash
# Default: 20s grace period, live preview window
python focus_guardian.py

# Nudge faster, and aim for a 50-minute focus block
python focus_guardian.py --grace 10 --goal 50

# Run without the preview window (lighter on the CPU)
python focus_guardian.py --no-window

# Use a second camera
python focus_guardian.py --camera 1
```

Press `q` (with the preview focused) or `Ctrl+C` to stop.

| Flag | Default | What it does |
| --- | --- | --- |
| `--grace` | `20` | Seconds away/looking-away before a nudge |
| `--goal` | none | Focus goal in minutes; session ends when reached |
| `--camera` | `0` | Which camera to use |
| `--no-window` | off | Run headless, no preview |
| `--no-gaze` | off | Presence only; skip eye/looking-away detection |
| `--no-calibrate` | off | Skip the startup eye-gaze calibration |
| `--gaze-threshold` | `0.25` | Eye-gaze strictness: `0.1` strict → `0.5` lenient |
| `--no-notify` | off | Terminal beep only; no desktop notifications |
| `--pomodoro` | off | Enable work/break cycles |
| `--work` | `25` | Pomodoro work minutes (implies `--pomodoro`) |
| `--break` | `5` | Pomodoro break minutes (implies `--pomodoro`) |
| `--stats` | — | Print the focus dashboard and exit |

## Ideas to extend it

These are good next steps if you want to keep growing the project:

- Estimate vertical gaze (looking *down* at a phone) as well as left/right.
- Multi-monitor / multi-face handling.
- Export the dashboard to an image or HTML chart.
- Sync the log across machines.

## Privacy

No frames are saved or sent anywhere. The only thing written to disk is
`focus_log.json`, which contains durations and counts — never images.

## License

MIT — do whatever you like, just keep the notice.
