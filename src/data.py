"""Data layer: NSL-KDD schema, real-dataset loader and a synthetic traffic generator.

The synthetic generator produces the *same 41 features* as NSL-KDD, so the whole
pipeline runs out-of-the-box (no download) and can be switched to the real
dataset by passing --data path/to/KDDTrain+.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL]
CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]

ATTACK_MAP = {
    # Denial of Service
    **dict.fromkeys(["back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
                     "apache2", "processtable", "udpstorm"], "dos"),
    # Probing / reconnaissance
    **dict.fromkeys(["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"], "probe"),
    # Remote to Local
    **dict.fromkeys(["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
                     "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
                     "snmpguess", "xlock", "xsnoop", "worm"], "r2l"),
    # User to Root
    **dict.fromkeys(["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel",
                     "ps", "sqlattack", "xterm"], "u2r"),
}


def to_category(label: str):
    label = str(label).strip().lower().rstrip(".")
    if label == "normal":
        return "normal"
    return ATTACK_MAP.get(label)  # None for unknown attack names


# --------------------------------------------------------------------------- #
# Real dataset (NSL-KDD / KDD Cup 99 style files)
# --------------------------------------------------------------------------- #
def load_nsl_kdd(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 43:            # 41 features + label + difficulty
        df.columns = COLUMNS + ["label", "difficulty"]
        df = df.drop(columns="difficulty")
    elif df.shape[1] == 42:          # 41 features + label
        df.columns = COLUMNS + ["label"]
    else:
        df = pd.read_csv(path)       # already has a header row
    df["label"] = df["label"].map(to_category)
    return df.dropna(subset=["label"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic traffic generator (NSL-KDD-like)
# --------------------------------------------------------------------------- #
_TCP_SERVICES = ["http", "ftp_data", "smtp", "private", "other", "telnet", "ftp", "ssh",
                 "pop_3", "auth", "finger", "imap4"]
_TCP_P = np.array([0.45, 0.12, 0.12, 0.03, 0.04, 0.03, 0.04, 0.04, 0.05, 0.03, 0.02, 0.03])
_TCP_P = _TCP_P / _TCP_P.sum()


def _base_row(rng: np.random.Generator) -> dict:
    """A benign connection record."""
    proto = rng.choice(["tcp", "udp", "icmp"], p=[0.80, 0.16, 0.04])
    if proto == "tcp":
        service = rng.choice(_TCP_SERVICES, p=_TCP_P)
    elif proto == "udp":
        service = rng.choice(["domain_u", "private", "ntp_u", "other"], p=[0.7, 0.15, 0.1, 0.05])
    else:
        service = "eco_i"
    flag = rng.choice(["SF", "REJ", "S0", "RSTO", "S1"], p=[0.93, 0.03, 0.01, 0.02, 0.01]) \
        if proto == "tcp" else "SF"
    count = int(np.clip(rng.poisson(7) + 1, 1, 511))
    srv_count = int(np.clip(rng.poisson(6) + 1, 1, 511))
    return dict(
        duration=float(rng.exponential(25) if rng.random() < 0.25 else 0),
        protocol_type=str(proto), service=str(service), flag=str(flag),
        src_bytes=float(rng.lognormal(5.6, 1.4)), dst_bytes=float(rng.lognormal(7.0, 1.8)),
        land=0, wrong_fragment=0, urgent=0, hot=int(rng.random() < 0.03),
        num_failed_logins=0, logged_in=int(proto == "tcp" and rng.random() < 0.75),
        num_compromised=0, root_shell=0, su_attempted=0, num_root=0,
        num_file_creations=0, num_shells=0, num_access_files=int(rng.random() < 0.01),
        num_outbound_cmds=0, is_host_login=0, is_guest_login=int(rng.random() < 0.02),
        count=count, srv_count=srv_count,
        serror_rate=float(rng.beta(1, 40)), srv_serror_rate=float(rng.beta(1, 40)),
        rerror_rate=float(rng.beta(1, 30)), srv_rerror_rate=float(rng.beta(1, 30)),
        same_srv_rate=float(rng.beta(9, 1)), diff_srv_rate=float(rng.beta(1, 18)),
        srv_diff_host_rate=float(rng.beta(1, 8)),
        dst_host_count=int(rng.integers(30, 256)), dst_host_srv_count=int(rng.integers(15, 256)),
        dst_host_same_srv_rate=float(rng.beta(8, 2)), dst_host_diff_srv_rate=float(rng.beta(1, 15)),
        dst_host_same_src_port_rate=float(rng.beta(1, 8)),
        dst_host_srv_diff_host_rate=float(rng.beta(1, 12)),
        dst_host_serror_rate=float(rng.beta(1, 40)), dst_host_srv_serror_rate=float(rng.beta(1, 40)),
        dst_host_rerror_rate=float(rng.beta(1, 30)), dst_host_srv_rerror_rate=float(rng.beta(1, 30)),
    )


def _dos_row(rng: np.random.Generator) -> dict:
    r = _base_row(rng)
    kind = rng.choice(["syn_flood", "smurf", "misc"], p=[0.60, 0.32, 0.08])
    if kind == "syn_flood":            # neptune-like
        r.update(protocol_type="tcp", service=str(rng.choice(["private", "http", "other", "smtp"])),
                 flag=str(rng.choice(["S0", "REJ"], p=[0.85, 0.15])), src_bytes=0.0, dst_bytes=0.0,
                 duration=0.0, logged_in=0, count=int(rng.integers(120, 512)),
                 srv_count=int(rng.integers(8, 30)), serror_rate=float(rng.beta(40, 1)),
                 srv_serror_rate=float(rng.beta(40, 1)), same_srv_rate=float(rng.beta(1, 8)),
                 diff_srv_rate=float(rng.beta(2, 8)), dst_host_count=255,
                 dst_host_srv_count=int(rng.integers(5, 25)),
                 dst_host_serror_rate=float(rng.beta(40, 1)),
                 dst_host_srv_serror_rate=float(rng.beta(40, 1)),
                 dst_host_same_srv_rate=float(rng.beta(1, 8)))
    elif kind == "smurf":              # ICMP echo amplification
        r.update(protocol_type="icmp", service="ecr_i", flag="SF",
                 src_bytes=float(rng.choice([520, 1032])), dst_bytes=0.0, duration=0.0,
                 logged_in=0, count=int(rng.integers(300, 512)),
                 srv_count=int(rng.integers(300, 512)), same_srv_rate=1.0, diff_srv_rate=0.0,
                 srv_diff_host_rate=0.0, dst_host_count=255, dst_host_srv_count=255,
                 dst_host_same_srv_rate=1.0, dst_host_diff_srv_rate=0.0,
                 dst_host_same_src_port_rate=1.0)
    else:                              # teardrop / pod / back / land
        r.update(wrong_fragment=int(rng.choice([1, 3])),
                 src_bytes=float(rng.choice([0, 1480, 54540])),
                 land=int(rng.random() < 0.15), hot=int(rng.integers(0, 3)),
                 count=int(rng.integers(1, 20)))
    return r


def _probe_row(rng: np.random.Generator) -> dict:
    r = _base_row(rng)
    kind = rng.choice(["portsweep", "ipsweep", "nmap_satan"], p=[0.40, 0.35, 0.25])
    common = dict(src_bytes=float(rng.choice([0, 8, 18, 28, 300])), dst_bytes=0.0,
                  duration=0.0, logged_in=0)
    if kind == "portsweep":
        r.update(**common, protocol_type="tcp",
                 service=str(rng.choice(_TCP_SERVICES + ["private"])),
                 flag=str(rng.choice(["REJ", "RSTR", "S0", "RSTO"])),
                 count=int(rng.integers(1, 6)), srv_count=int(rng.integers(1, 4)),
                 rerror_rate=float(rng.beta(6, 2)), same_srv_rate=float(rng.beta(1, 6)),
                 diff_srv_rate=float(rng.beta(6, 2)),
                 dst_host_count=int(rng.integers(1, 60)),
                 dst_host_srv_count=int(rng.integers(1, 15)),
                 dst_host_diff_srv_rate=float(rng.beta(6, 2)),
                 dst_host_rerror_rate=float(rng.beta(5, 2)))
    elif kind == "ipsweep":
        r.update(**common, protocol_type="icmp", service="eco_i", flag="SF",
                 count=int(rng.integers(1, 6)), srv_count=int(rng.integers(1, 20)),
                 same_srv_rate=1.0, diff_srv_rate=0.0,
                 srv_diff_host_rate=float(rng.beta(6, 2)),
                 dst_host_count=int(rng.integers(1, 30)),
                 dst_host_srv_count=int(rng.integers(1, 30)),
                 dst_host_srv_diff_host_rate=float(rng.beta(4, 3)))
    else:
        r.update(**common, protocol_type=str(rng.choice(["tcp", "udp"])),
                 service=str(rng.choice(["private", "other", "http", "domain_u"])),
                 flag=str(rng.choice(["SF", "S0", "REJ", "RSTR"])),
                 count=int(rng.integers(1, 15)),
                 diff_srv_rate=float(rng.beta(4, 3)), same_srv_rate=float(rng.beta(2, 4)),
                 dst_host_diff_srv_rate=float(rng.beta(4, 3)),
                 dst_host_same_src_port_rate=float(rng.beta(4, 2)),
                 dst_host_rerror_rate=float(rng.beta(2, 4)))
    return r


def _r2l_row(rng: np.random.Generator) -> dict:
    r = _base_row(rng)
    if rng.random() < 0.5:             # password guessing
        r.update(protocol_type="tcp",
                 service=str(rng.choice(["ftp", "telnet", "pop_3", "imap4", "ssh"])),
                 flag="SF", num_failed_logins=int(rng.integers(1, 6)),
                 logged_in=int(rng.random() < 0.3), duration=float(rng.exponential(5)),
                 src_bytes=float(rng.integers(50, 200)), dst_bytes=float(rng.integers(0, 500)),
                 count=int(rng.integers(1, 6)),
                 dst_host_same_src_port_rate=float(rng.beta(4, 2)))
    else:                              # warez / ftp abuse
        r.update(protocol_type="tcp", service=str(rng.choice(["ftp_data", "ftp"])), flag="SF",
                 logged_in=1, is_guest_login=1, hot=int(rng.integers(1, 30)),
                 src_bytes=float(rng.lognormal(9, 1.5)), dst_bytes=float(rng.integers(0, 300)),
                 duration=float(rng.exponential(300)),
                 num_file_creations=int(rng.integers(0, 3)))
    return r


def _u2r_row(rng: np.random.Generator) -> dict:
    r = _base_row(rng)
    r.update(protocol_type="tcp", service=str(rng.choice(["telnet", "ftp_data", "other", "ssh"])),
             flag="SF", logged_in=1, duration=float(rng.exponential(120)),
             hot=int(rng.integers(0, 6)), num_compromised=int(rng.integers(0, 4)),
             root_shell=int(rng.random() < 0.7), su_attempted=int(rng.random() < 0.25),
             num_root=int(rng.integers(0, 8)), num_file_creations=int(rng.integers(0, 6)),
             num_shells=int(rng.random() < 0.4), num_access_files=int(rng.integers(0, 3)),
             src_bytes=float(rng.integers(0, 4000)), dst_bytes=float(rng.integers(0, 6000)))
    return r


_GENERATORS = {"normal": _base_row, "dos": _dos_row, "probe": _probe_row,
               "r2l": _r2l_row, "u2r": _u2r_row}
DEFAULT_MIX = {"normal": 0.53, "dos": 0.30, "probe": 0.10, "r2l": 0.05, "u2r": 0.02}


def generate_synthetic(n: int = 30000, seed: int = 42, mix: dict | None = None,
                       stealth: float = 0.05) -> pd.DataFrame:
    """Generate `n` labelled connection records.

    `stealth` = fraction of DoS/Probe records made to look benign (R2L/U2R use
    higher built-in rates) so the task is not trivially separable.
    """
    rng = np.random.default_rng(seed)
    mix = mix or DEFAULT_MIX
    p = np.array(list(mix.values()), dtype=float)
    labels = rng.choice(list(mix), size=n, p=p / p.sum())
    rows = []
    for lab in labels:
        row = _GENERATORS[lab](rng)
        # low-and-slow attacks that mimic benign traffic (R2L/U2R are inherently stealthy)
        rate = {"dos": stealth, "probe": stealth, "r2l": 0.30, "u2r": 0.35}.get(lab, 0.0)
        if rng.random() < rate:
            row = _base_row(rng)
        row["label"] = lab
        rows.append(row)
    return pd.DataFrame(rows)[COLUMNS + ["label"]]


def load_dataset(path: str | Path | None = None, n_synthetic: int = 30000, seed: int = 42):
    """Return (dataframe, source_description)."""
    if path and Path(path).exists():
        return load_nsl_kdd(path), f"real dataset: {path}"
    return generate_synthetic(n_synthetic, seed), \
        f"synthetic NSL-KDD-like traffic (n={n_synthetic})"
