# Fitbit Google Takeout → CSV (Activities + Sleep)

A small **Streamlit** web app that converts a **Google Takeout** ZIP export of your **Fitbit** data into a single CSV file containing two sections:

- **Activities** (daily aggregates: steps, calories, distance, activity minutes, …)
- **Sleep** (one row per sleep log)

The output format is intentionally “dashboard-style”: one CSV with an `Activities` block followed by a `Sleep` block.

> Not affiliated with Fitbit, Google, or Alphabet.

---

## What this app does

- Accepts a Google Takeout `.zip` you download from takeout.google.com
- Locates the Fitbit data folder inside the archive
- Parses common Fitbit Takeout JSON files (e.g., `steps*.json`, `calories*.json`, `distance*.json`, `sleep*.json`, …)
- Aggregates time-series metrics to **daily totals**
- Produces one CSV you can download, plus in-app previews of the activity and sleep tables

---

## Quick start

### 1) Install

```bash
git clone https://github.com/zionwzz/Fitbit-Google-Takeout-zip.file-to-Fitbit-Dashboard-CSV-Activity-Sleep-.git
cd Fitbit-Google-Takeout-zip.file-to-Fitbit-Dashboard-CSV-Activity-Sleep-

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\\Scripts\\Activate.ps1

pip install -r requirement.txt
```

> Note: the repo currently uses **`requirement.txt`** (singular) instead of the more common `requirements.txt`.

### 2) Run the Streamlit app

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

---

## Getting your Fitbit data ZIP (Google Takeout)

1. Go to **Google Takeout** (takeout.google.com)
2. Deselect all products, then select **Fitbit**
3. Click **Next step**
4. Choose export format **.zip**
5. Create the export and download the ZIP when it’s ready

### If Google gives you multiple ZIP files

This app accepts **one** ZIP upload.

If Takeout splits your export into multiple ZIPs, combine them into a single ZIP by:

1. Extracting all ZIP parts into **one** folder (so the internal `Takeout/` structure merges)
2. Re-zipping that combined folder into a single ZIP

---

## Input expectations

Inside the uploaded ZIP, the app expects Fitbit “Global Export Data” JSON files under one of these common paths:

- `Fitbit/Global Export Data`
- `Fitbit/GlobalExportData`

(There are a few fallbacks in the code to handle minor naming/spacing differences.)

---

## Using the app

1. Enter a **Participant ID** (used only to name the downloaded file)
2. Pick the study date range:
   - **Day1 Set up / Start Date**
   - **Box Returned Date**
3. Choose whether to:
   - ✅ **Keep only intersection date range across domains** (recommended for clean day-by-day datasets)
4. Upload your Takeout `.zip`
5. Click **Process**
6. Preview the generated tables and click **Download CSV**

The downloaded filename is:

```
Fitbit_<ParticipantID>_<start-date>_<end-date>.csv
```

---

## Output format

The generated CSV has two labeled sections:

```text
Activities
<Date/Calories/... table>

Sleep
<Sleep table>
```

### Activities columns

| Column | Meaning | Units / Notes |
|---|---|---|
| `Date` | Day | `YYYY-MM-DD` |
| `Calories Burned` | Total calories burned that day | kcal |
| `Steps` | Total steps that day | count |
| `Distance` | Total distance that day | **miles** (converted from cm) |
| `Floors` | Floors climbed | currently always `0` |
| `Minutes Sedentary` | Sedentary minutes | minutes |
| `Minutes Lightly Active` | Light activity minutes | minutes |
| `Minutes Fairly Active` | Moderate activity minutes | minutes |
| `Minutes Very Active` | Vigorous activity minutes | minutes |
| `Activity Calories` | Estimated active calories | kcal (see heuristic below) |

### Sleep columns

| Column | Meaning | Units / Notes |
|---|---|---|
| `Start Time` | Sleep start | formatted like `YYYY-MM-DD h:MMAM/PM` |
| `End Time` | Sleep end | same formatting |
| `Minutes Asleep` | Minutes asleep | minutes |
| `Minutes Awake` | Minutes awake during the session | minutes |
| `Number of Awakenings` | Wakes/restless count | count (varies by Takeout schema) |
| `Time in Bed` | Total time in bed | minutes |
| `Minutes REM Sleep` | REM minutes | staged logs only; otherwise `N/A` |
| `Minutes Light Sleep` | Light sleep minutes | staged logs only; otherwise `N/A` |
| `Minutes Deep Sleep` | Deep sleep minutes | staged logs only; otherwise `N/A` |

---

## Notes on calculations and heuristics

### Distance conversion

Fitbit Takeout `distance*.json` values are treated as **centimeters**.

The app converts to miles using:

- `cm → meters` by dividing by `100`
- `meters → miles` by dividing by `1609.344`

### Activity calories estimation

Fitbit exports a minute-level (or interval-level) calories time series. The app estimates **active** calories by:

1. Summing all per-interval calories for a day → `total`
2. Taking the minimum per-interval value for that day as a baseline → `baseline`
3. Estimating non-active calories as `baseline * number_of_intervals`
4. `activity_calories = max(total - baseline * number_of_intervals, 0)`

This is a pragmatic approximation that assumes the minimum observed interval represents resting/basal burn.

**Limitations:** if your calories series is not minute-level, has missing intervals, or uses a different schema, this estimate may be off.

### “Intersection date range” option

When enabled, the app restricts output to the date range where **all parsed domains overlap** (steps, calories, distance, activity minutes, and sleep).

This is useful when you need a dataset with no partial-day holes, but it can shorten the output if one domain starts later or ends earlier.

---

## Troubleshooting

### “Fitbit/Global Export Data folder not found in this zip.”

- Ensure you uploaded the **Google Takeout ZIP**, not a nested folder or a partial ZIP.
- If you got multiple ZIP parts from Takeout, merge them into a single ZIP (see above).

### “No usable activity or sleep records found…”

- Your selected date range may not overlap with available data.
- Some exports may not include the JSON files this app expects.

### Upload too large

This repo includes `.streamlit/config.toml` with:

```toml
[server]
maxUploadSize = 500
```

Increase it if your Takeout ZIP is larger.

---

## Privacy

By default, everything runs **locally** on your machine.

If you deploy this app to a hosted environment (Streamlit Community Cloud, a server, etc.), be mindful that Fitbit exports can contain sensitive personal health data.

---

## Development notes

- The repo includes a `.devcontainer/` for Codespaces / VS Code Dev Containers.
- Core conversion logic lives in pure Python functions so it can be reused outside Streamlit.

---

## License

No license file is currently included in this repository. If you intend others to reuse or modify it, consider adding an explicit open-source license.
