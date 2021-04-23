import datetime as dt


def datetime_to_json(d: dt.datetime) -> str:
    # Regular Python ISO format: 2021-04-23T09:05:16.157178+00:00
    # ECMA-262 format:           2021-04-23T09:05:16.157Z
    s = d.isoformat()
    if s.endswith("+00:00"):
        # Replace explicit zero offset by UTC marker
        s = s[:-6] + "Z"
    if d.microsecond:
        # Remove 3 digits of microsecond precision
        s = s[:23] + s[26:]

    return s


def json_to_datetime(s: str) -> dt.datetime:
    # Regular Python ISO format: 2021-04-23T09:05:16.157178+00:00
    # ECMA-262 format:           2021-04-23T09:05:16.157Z
    if s.endswith("Z"):
        # Restore explicit zero offset
        s = s[:-1] + "+00:00"

    return dt.datetime.fromisoformat(s)


def time_to_json(t: dt.time) -> str:
    # Regular Python ISO format: 09:05:16.157178
    # ECMA-262 format:           09:05:16.157
    if t.utcoffset() is not None:
        raise ValueError("Timezone-aware times cannot be represented in JSON")

    s = t.isoformat()
    if t.microsecond:
        s = s[:12]

    return s


def json_to_time(s: str) -> dt.time:
    return dt.time.fromisoformat(s)

