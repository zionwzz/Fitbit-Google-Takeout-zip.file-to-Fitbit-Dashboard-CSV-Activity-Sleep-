# app.py
import io
import json
import csv
import zipfile
import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------
# Core parsing utilities
# -----------------------
def _is_macosx_path(p: Path) -> bool:
    return "__MACOSX" in p.parts


def safe_load_json(fp: Path) -> Optional[Any]:
    try:
        txt = fp.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        obj = json.loads(txt)
        return obj if obj else None
    except Exception:
        return None


def parse_fitbit_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, format="%m/%d/%y %H:%M:%S", errors="coerce")
    if dt.isna().all():
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    miss = dt.isna()
    if miss.any():
        dt.loc[miss] = pd.to_datetime(series[miss], errors="coerce", infer_datetime_format=True)
    return dt


def cm_to_miles(s: pd.Series) -> pd.Series:
    return (s / 100.0) / 1609.344


def find_global_export_data_folder(extract_root: Path) -> Path:
    extract_root = extract_root.resolve()
    candidates = [
        p for p in (
            list(extract_root.rglob("Fitbit/Global Export Data")) +
            list(extract_root.rglob("Fitbit/GlobalExportData"))
        )
        if p.is_dir() and not _is_macosx_path(p)
    ]

    if not candidates:
        for p in extract_root.rglob("*"):
            if not p.is_dir() or _is_macosx_path(p):
                continue
            nm = p.name.lower().replace(" ", "")
            if nm == "globalexportdata" and any(pp.name.lower() == "fitbit" for pp in p.parents):
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError("Fitbit/Global Export Data folder not found in this zip.")

    def score(p: Path) -> int:
        return len([x for x in p.glob("*.json") if x.is_file() and not _is_macosx_path(x)])

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def collect_prefix_files(folder: Path, prefix: str) -> List[Path]:
    folder = folder.resolve()
    direct = [fp for fp in folder.glob(f"{prefix}*.json") if fp.is_file() and not _is_macosx_path(fp)]
    if direct:
        return sorted(set(direct))
    deep = [fp for fp in folder.rglob(f"{prefix}*.json") if fp.is_file() and not _is_macosx_path(fp)]
    return sorted(set(deep))


def collect_sleep_files(folder: Path) -> List[Path]:
    folder = folder.resolve()
    direct = [fp for fp in folder.glob("sleep*.json") if fp.is_file() and not _is_macosx_path(fp)]
    direct += [fp for fp in folder.glob("Sleep*.json") if fp.is_file() and not _is_macosx_path(fp)]
    if direct:
        return sorted(set(direct))
    deep = [fp for fp in folder.rglob("sleep*.json") if fp.is_file() and not _is_macosx_path(fp)]
    deep += [fp for fp in folder.rglob("Sleep*.json") if fp.is_file() and not _is_macosx_path(fp)]
    return sorted(set(deep))


def metric_daily_sum(
    files: List[Path],
    out_col: str,
    round_int: bool = False,
    value_transform=None,
) -> pd.DataFrame:
    dfs = []
    for fp in files:
        obj = safe_load_json(fp)
        if obj is None:
            continue
        df = pd.DataFrame(obj) if isinstance(obj, list) else pd.DataFrame([obj])
        if df.empty or "dateTime" not in df.columns or "value" not in df.columns:
            continue

        dt = parse_fitbit_datetime(df["dateTime"])
        val = pd.to_numeric(df["value"], errors="coerce")
        if value_transform is not None:
            val = value_transform(val)

        tmp = pd.DataFrame({"date": dt.dt.date, "value": val}).dropna()
        if tmp.empty:
            continue

        if round_int:
            tmp["value"] = tmp["value"].round().astype(int)

        daily = tmp.groupby("date", as_index=False)["value"].sum()
        daily.rename(columns={"value": out_col}, inplace=True)
        dfs.append(daily)

    if not dfs:
        return pd.DataFrame(columns=["date", out_col])

    return pd.concat(dfs, ignore_index=True).groupby("date", as_index=False).sum()


def build_daily_calories_and_activity(global_folder: Path) -> pd.DataFrame:
    files = collect_prefix_files(global_folder, "calories")
    if not files:
        return pd.DataFrame(columns=["date", "Calories Burned", "Activity Calories"])

    parts = []
    for fp in files:
        obj = safe_load_json(fp)
        if obj is None:
            continue
        df = pd.DataFrame(obj) if isinstance(obj, list) else pd.DataFrame([obj])
        if df.empty or "dateTime" not in df.columns or "value" not in df.columns:
            continue
        dt = parse_fitbit_datetime(df["dateTime"])
        kcal = pd.to_numeric(df["value"], errors="coerce")
        tmp = pd.DataFrame({"date": dt.dt.date, "kcal": kcal}).dropna()
        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        return pd.DataFrame(columns=["date", "Calories Burned", "Activity Calories"])

    all_min = pd.concat(parts, ignore_index=True)
    g = all_min.groupby("date")["kcal"]
    total = g.sum()
    baseline = g.min()
    n = g.count()
    activity = (total - baseline * n).clip(lower=0)

    out = pd.DataFrame(
        {"date": total.index, "Calories Burned": total.values, "Activity Calories": activity.values}
    )
    out["Calories Burned"] = out["Calories Burned"].round().astype(int)
    out["Activity Calories"] = out["Activity Calories"].round().astype(int)
    return out


def _fmt_dt_sleep(dt: pd.Timestamp) -> str:
    s = dt.strftime("%Y-%m-%d %I:%M%p")
    return s.replace(" 0", " ")


def _extract_sleep_row(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    start = pd.to_datetime(ev.get("startTime"), errors="coerce")
    end = pd.to_datetime(ev.get("endTime"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None

    levels = ev.get("levels") or {}
    summary = (levels.get("summary") or {})

    wake_count = (summary.get("wake") or {}).get("count", None)
    if wake_count is None:
        awake_count = int((summary.get("awake") or {}).get("count") or 0)
        restless_count = int((summary.get("restless") or {}).get("count") or 0)
        num_awakenings = awake_count + restless_count
    else:
        num_awakenings = int(wake_count)

    # ----- staged vs classic handling -----
    sleep_type = str(ev.get("type") or "").strip().lower()
    is_staged = sleep_type in {"stages", "staged"}

    # If type missing, infer from keys
    if not sleep_type:
        is_staged = any(k in summary for k in ["rem", "light", "deep"])

    if is_staged:
        rem_min = int((summary.get("rem") or {}).get("minutes") or 0)
        light_min = int((summary.get("light") or {}).get("minutes") or 0)
        deep_min = int((summary.get("deep") or {}).get("minutes") or 0)
    else:
        rem_min = "N/A"
        light_min = "N/A"
        deep_min = "N/A"

    return {
        "Start Time": _fmt_dt_sleep(start),
        "End Time": _fmt_dt_sleep(end),
        "Minutes Asleep": int(ev.get("minutesAsleep") or 0),
        "Minutes Awake": int(ev.get("minutesAwake") or 0),
        "Number of Awakenings": int(num_awakenings),
        "Time in Bed": int(ev.get("timeInBed") or 0),
        "Minutes REM Sleep": rem_min,
        "Minutes Light Sleep": light_min,
        "Minutes Deep Sleep": deep_min,
        "_start_dt": start,
        "_logId": ev.get("logId", None),
    }


def build_sleep_table(global_folder: Path) -> pd.DataFrame:
    out_cols = [
        "Start Time", "End Time", "Minutes Asleep", "Minutes Awake",
        "Number of Awakenings", "Time in Bed",
        "Minutes REM Sleep", "Minutes Light Sleep", "Minutes Deep Sleep"
    ]

    rows: List[Dict[str, Any]] = []
    for fp in collect_sleep_files(global_folder):
        obj = safe_load_json(fp)
        if obj is None:
            continue
        events = obj if isinstance(obj, list) else [obj]
        for ev in events:
            if not isinstance(ev, dict) or "startTime" not in ev or "endTime" not in ev:
                continue
            row = _extract_sleep_row(ev)
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=out_cols)

    df = pd.DataFrame(rows)
    if df["_logId"].notna().any():
        df = df.drop_duplicates(subset=["_logId"], keep="first")
    else:
        df = df.drop_duplicates(subset=["Start Time", "End Time"], keep="first")

    df = df.sort_values("_start_dt", ascending=True).drop(columns=["_logId"])
    return df[out_cols + ["_start_dt"]]


def _range_from_daily(df: pd.DataFrame, date_col: str = "date") -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if df is None or df.empty or date_col not in df.columns:
        return None
    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if d.empty:
        return None
    return (d.min().normalize(), d.max().normalize())


def _intersect_ranges(ranges: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not ranges:
        return None
    start = max(r[0] for r in ranges)
    end = min(r[1] for r in ranges)
    return None if start > end else (start, end)


def _filter_daily_by_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, date_col: str = "date") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    return df.loc[(d >= start) & (d <= end)].copy()


def build_outputs(
    global_folder: Path,
    intersect_dates: bool = True,
    user_start: Optional[pd.Timestamp] = None,
    user_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Tuple[pd.Timestamp, pd.Timestamp]]]:

    steps = metric_daily_sum(collect_prefix_files(global_folder, "steps"), "Steps", round_int=True)
    cal_act = build_daily_calories_and_activity(global_folder)
    dist = metric_daily_sum(collect_prefix_files(global_folder, "distance"), "Distance", value_transform=cm_to_miles)

    sed = metric_daily_sum(collect_prefix_files(global_folder, "sedentary_minutes"), "Minutes Sedentary", round_int=True)
    light = metric_daily_sum(collect_prefix_files(global_folder, "lightly_active_minutes"), "Minutes Lightly Active", round_int=True)
    mod = metric_daily_sum(collect_prefix_files(global_folder, "moderately_active_minutes"), "Minutes Fairly Active", round_int=True)
    vig = metric_daily_sum(collect_prefix_files(global_folder, "very_active_minutes"), "Minutes Very Active", round_int=True)

    sleep_raw = build_sleep_table(global_folder)

    # Normalize user range (inclusive)
    user_range = None
    if user_start is not None and user_end is not None:
        us = pd.to_datetime(user_start, errors="coerce")
        ue = pd.to_datetime(user_end, errors="coerce")
        if not pd.isna(us) and not pd.isna(ue):
            user_range = (us.normalize(), ue.normalize())

    date_range = None

    if intersect_dates:
        ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for d in [steps, cal_act, dist, sed, light, mod, vig]:
            r = _range_from_daily(d, "date")
            if r:
                ranges.append(r)

        if not sleep_raw.empty:
            sd = sleep_raw["_start_dt"].dropna()
            if not sd.empty:
                ranges.append((sd.min().normalize(), sd.max().normalize()))

        date_range = _intersect_ranges(ranges)
        if date_range is None:
            return pd.DataFrame(), pd.DataFrame(), None

        # intersect with user range if provided
        if user_range is not None:
            date_range = _intersect_ranges([date_range, user_range])
            if date_range is None:
                return pd.DataFrame(), pd.DataFrame(), None

        start, end = date_range
        steps = _filter_daily_by_range(steps, start, end)
        cal_act = _filter_daily_by_range(cal_act, start, end)
        dist = _filter_daily_by_range(dist, start, end)
        sed = _filter_daily_by_range(sed, start, end)
        light = _filter_daily_by_range(light, start, end)
        mod = _filter_daily_by_range(mod, start, end)
        vig = _filter_daily_by_range(vig, start, end)

        if not sleep_raw.empty:
            sd = sleep_raw["_start_dt"].dt.normalize()
            sleep_raw = sleep_raw.loc[(sd >= start) & (sd <= end)].copy()

    else:
        # only apply user range (if provided)
        if user_range is not None:
            start, end = user_range
            steps = _filter_daily_by_range(steps, start, end)
            cal_act = _filter_daily_by_range(cal_act, start, end)
            dist = _filter_daily_by_range(dist, start, end)
            sed = _filter_daily_by_range(sed, start, end)
            light = _filter_daily_by_range(light, start, end)
            mod = _filter_daily_by_range(mod, start, end)
            vig = _filter_daily_by_range(vig, start, end)

            if not sleep_raw.empty:
                sd = sleep_raw["_start_dt"].dt.normalize()
                sleep_raw = sleep_raw.loc[(sd >= start) & (sd <= end)].copy()

        date_range = user_range

    # Combine activity
    dfs = [steps, cal_act, dist, sed, light, mod, vig]
    combined = None
    for d in dfs:
        combined = d if combined is None else pd.merge(combined, d, on="date", how="outer")

    garmin_cols = [
        "Date", "Calories Burned", "Steps", "Distance", "Floors",
        "Minutes Sedentary", "Minutes Lightly Active", "Minutes Fairly Active",
        "Minutes Very Active", "Activity Calories"
    ]

    if combined is None or combined.empty:
        activity_df = pd.DataFrame(columns=garmin_cols)
    else:
        combined.fillna(0, inplace=True)
        combined["Date"] = combined["date"].astype(str)
        combined["Floors"] = 0

        for c in ["Calories Burned", "Steps", "Distance", "Activity Calories",
                  "Minutes Sedentary", "Minutes Lightly Active", "Minutes Fairly Active",
                  "Minutes Very Active", "Floors"]:
            if c not in combined.columns:
                combined[c] = 0

        for c in ["Steps", "Floors", "Minutes Sedentary", "Minutes Lightly Active",
                  "Minutes Fairly Active", "Minutes Very Active", "Activity Calories", "Calories Burned"]:
            combined[c] = combined[c].round().fillna(0).astype(int)

        combined["Distance"] = combined["Distance"].astype(float)
        activity_df = combined[garmin_cols].sort_values("Date")

    # Sleep output
    sleep_cols = [
        "Start Time", "End Time", "Minutes Asleep", "Minutes Awake",
        "Number of Awakenings", "Time in Bed",
        "Minutes REM Sleep", "Minutes Light Sleep", "Minutes Deep Sleep"
    ]
    if sleep_raw.empty:
        sleep_df = pd.DataFrame(columns=sleep_cols)
    else:
        sleep_df = (
            sleep_raw.sort_values("_start_dt", ascending=True)
                    .drop(columns=["_start_dt"])
                    .reset_index(drop=True)
        )

    return activity_df, sleep_df, date_range


def write_combined_csv_bytes(activity_df: pd.DataFrame, sleep_df: pd.DataFrame) -> bytes:
    if activity_df.empty and sleep_df.empty:
        return b""

    sleep_cols = [
        "Start Time", "End Time", "Minutes Asleep", "Minutes Awake",
        "Number of Awakenings", "Time in Bed",
        "Minutes REM Sleep", "Minutes Light Sleep", "Minutes Deep Sleep"
    ]

    buf = io.StringIO()
    buf.write("Activities\n")
    activity_df.to_csv(buf, index=False, float_format="%.2f")

    buf.write("\nSleep\n")
    buf.write(",".join(sleep_cols) + "\n")
    w = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
    for _, r in sleep_df.iterrows():
        w.writerow([r.get(c, "") for c in sleep_cols])

    return buf.getvalue().encode("utf-8")


def convert_takeout_zip_bytes(
    zip_bytes: bytes,
    intersect_dates: bool = True,
    user_start: Optional[pd.Timestamp] = None,
    user_end: Optional[pd.Timestamp] = None,
) -> Tuple[bytes, pd.DataFrame, pd.DataFrame, Optional[Tuple[pd.Timestamp, pd.Timestamp]]]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            z.extractall(root)
        global_folder = find_global_export_data_folder(root)
        activity_df, sleep_df, date_range = build_outputs(
            global_folder,
            intersect_dates=intersect_dates,
            user_start=user_start,
            user_end=user_end,
        )
        out_bytes = write_combined_csv_bytes(activity_df, sleep_df)
        return out_bytes, activity_df, sleep_df, date_range


def _sanitize_filename_part(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s or "UNKNOWN"


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Fitbit Takeout → CSV", layout="centered")
st.title("Fitbit Takeout → CSV")

with st.form("inputs", clear_on_submit=False):
    participant_id = st.text_input("Participant ID", value="")
    day1 = st.date_input("Day1 Set up / Start Date")
    returned = st.date_input("Box Returned Date")
    intersect = st.checkbox("Keep only intersection date range across domains", value=True)
    uploaded = st.file_uploader("Upload Google Takeout zip", type=["zip"])
    submitted = st.form_submit_button("Process")

if submitted:
    if uploaded is None:
        st.error("Please upload a Google Takeout zip.")
        st.stop()

    if not participant_id.strip():
        st.error("Please enter Participant ID.")
        st.stop()

    if day1 > returned:
        st.error("Start Date must be on or before Returned Date.")
        st.stop()

    try:
        user_start = pd.Timestamp(day1)
        user_end = pd.Timestamp(returned)

        out_bytes, act_df, slp_df, dr = convert_takeout_zip_bytes(
            uploaded.getvalue(),
            intersect_dates=intersect,
            user_start=user_start,
            user_end=user_end,
        )

        if out_bytes == b"":
            st.warning("No usable activity or sleep records found in the selected date range.")
            st.stop()

        if dr is not None:
            st.caption(f"Output date range: {dr[0].date()} to {dr[1].date()}")
        else:
            st.caption(f"Filtered by user range: {day1} to {returned}")

        with st.expander("Preview: Activities", expanded=True):
            st.dataframe(act_df.head(50), use_container_width=True)

        with st.expander("Preview: Sleep", expanded=False):
            st.dataframe(slp_df.head(50), use_container_width=True)

        pid = _sanitize_filename_part(participant_id)
        file_name = f"Fitbit_{pid}_{day1.isoformat()}_{returned.isoformat()}.csv"

        st.download_button(
            "Download CSV",
            data=out_bytes,
            file_name=file_name,
            mime="text/csv",
        )

    except Exception as e:
        st.error(str(e))
