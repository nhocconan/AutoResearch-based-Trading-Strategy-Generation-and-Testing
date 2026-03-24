"""
santo-green-v22-2am-night-edge
Session-based EMA crossover strategy for overnight trading (2AM-3AM ET)
"""

import numpy as np
import pandas as pd
from datetime import datetime, time

name = "santo-green-v22-2am-night-edge"
timeframe = "1m"
leverage = 1

# Strategy parameters (from Pine inputs)
PARAMS = {
    "ema_fast": 5,
    "ema_slow": 13,
    "rsi_len": 14,
    "rsi_ob": 60.0,
    "rsi_os": 40.0,
    "short_bias": True,
    "tp_usd": 30.0,
    "sl_usd": 150.0,
    "trail_usd": 75.0,
    "use_trail": False,
    "max_trades": 3,
    "daily_loss": 300.0,
    "session_start_hour": 2,
    "session_start_min": 0,
    "session_end_hour": 3,
    "session_end_min": 0,
}

# NQ contract specs (CME Mini Nasdaq-100)
# Point value: $5 per point, Min tick: 0.25 points = $1.25
NQ_POINT_VALUE = 5.0
NQ_MIN_TICK = 0.25


def _ema(series: np.ndarray, length: int) -> np.ndarray:
    """Calculate EMA using pandas for consistency."""
    return pd.Series(series).ewm(span=length, adjust=False).mean().values


def _rsi(series: np.ndarray, length: int) -> np.ndarray:
    """Calculate RSI."""
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=length, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=length, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _convert_to_et(dt):
    """Convert datetime to Eastern Time using pandas."""
    try:
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts = ts.tz_convert("America/New_York")
        return ts
    except Exception:
        return pd.Timestamp(dt)


def _in_session(dt, start_hour: int, start_min: int,
                end_hour: int, end_min: int) -> bool:
    """Check if datetime is within session (Eastern Time)."""
    try:
        dt_et = _convert_to_et(dt)
        t_now = dt_et.hour * 60 + dt_et.minute
        sess_open = start_hour * 60 + start_min
        sess_close = end_hour * 60 + end_min
        return sess_open <= t_now <= sess_close
    except Exception:
        return True


def _day_ok(dt) -> bool:
    """Check if day is Mon-Fri."""
    try:
        dt_et = _convert_to_et(dt)
        return dt_et.weekday() < 5
    except Exception:
        return True


def _get_date_key(dt) -> str:
    """Get date key for daily counters."""
    try:
        dt_et = _convert_to_et(dt)
        return dt_et.strftime("%Y-%m-%d")
    except Exception:
        return str(pd.Timestamp(dt).date())


def _at_session_close(dt, end_hour: int, end_min: int) -> bool:
    """Check if at or past session close."""
    try:
        dt_et = _convert_to_et(dt)
        t_now = dt_et.hour * 60 + dt_et.minute
        sess_close = end_hour * 60 + end_min
        return t_now >= sess_close
    except Exception:
        return False


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Generate trading signals for the strategy.
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume]
    
    Returns:
        numpy.ndarray of position intent: 1=long, -1=short, 0=flat
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    if n < max(PARAMS["ema_slow"], PARAMS["rsi_len"]) + 5:
        return signals
    
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    open_time = prices["open_time"].values
    
    ema_f = _ema(close, PARAMS["ema_fast"])
    ema_s = _ema(close, PARAMS["ema_slow"])
    rsi = _rsi(close, PARAMS["rsi_len"])
    
    tp_ticks = int(round(PARAMS["tp_usd"] / NQ_POINT_VALUE / NQ_MIN_TICK))
    sl_ticks = int(round(PARAMS["sl_usd"] / NQ_POINT_VALUE / NQ_MIN_TICK))
    trail_ticks = int(round(PARAMS["trail_usd"] / NQ_POINT_VALUE / NQ_MIN_TICK))
    
    tp_points = tp_ticks * NQ_MIN_TICK
    sl_points = sl_ticks * NQ_MIN_TICK
    trail_points = trail_ticks * NQ_MIN_TICK if PARAMS["use_trail"] else None
    
    cross_up = np.zeros(n, dtype=bool)
    cross_dn = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if ema_f[i-1] <= ema_s[i-1] and ema_f[i] > ema_s[i]:
            cross_up[i] = True
        if ema_f[i-1] >= ema_s[i-1] and ema_f[i] < ema_s[i]:
            cross_dn[i] = True
    
    state = {
        "position": 0,
        "entry_price": 0.0,
        "trades_today": 0,
        "daily_pnl": 0.0,
        "last_date": None,
        "highest_since_entry": 0.0,
        "lowest_since_entry": 0.0,
    }
    
    for i in range(n):
        dt = open_time[i]
        
        cur_date = _get_date_key(dt)
        if state["last_date"] != cur_date:
            state["trades_today"] = 0
            state["daily_pnl"] = 0.0
            state["last_date"] = cur_date
        
        in_sess = _in_session(dt, PARAMS["session_start_hour"],
                              PARAMS["session_start_min"],
                              PARAMS["session_end_hour"],
                              PARAMS["session_end_min"])
        
        day_valid = _day_ok(dt)
        at_close = _at_session_close(dt, PARAMS["session_end_hour"],
                                     PARAMS["session_end_min"])
        
        trade_limit_ok = state["trades_today"] < PARAMS["max_trades"]
        loss_limit_ok = state["daily_pnl"] > -abs(PARAMS["daily_loss"])
        
        can_trade = in_sess and day_valid and trade_limit_ok and loss_limit_ok
        
        long_sig = cross_up[i] and (rsi[i] < PARAMS["rsi_ob"]) and (rsi[i] > 40)
        short_sig = cross_dn[i] and (rsi[i] > PARAMS["rsi_os"]) and (rsi[i] < 60)
        
        long_allowed = (not PARAMS["short_bias"]) or (rsi[i] < 50)
        short_allowed = True
        
        if state["position"] == 0 and can_trade and not at_close:
            if long_sig and long_allowed:
                state["position"] = 1
                state["entry_price"] = close[i]
                state["highest_since_entry"] = close[i]
                state["lowest_since_entry"] = close[i]
                state["trades_today"] += 1
                signals[i] = 1
            elif short_sig and short_allowed:
                state["position"] = -1
                state["entry_price"] = close[i]
                state["highest_since_entry"] = close[i]
                state["lowest_since_entry"] = close[i]
                state["trades_today"] += 1
                signals[i] = -1
        
        elif state["position"] != 0:
            exit_triggered = False
            
            if state["position"] == 1:
                state["highest_since_entry"] = max(state["highest_since_entry"], high[i])
                
                if low[i] <= state["entry_price"] - sl_points:
                    state["daily_pnl"] -= sl_points * NQ_POINT_VALUE
                    exit_triggered = True
                elif high[i] >= state["entry_price"] + tp_points:
                    state["daily_pnl"] += tp_points * NQ_POINT_VALUE
                    exit_triggered = True
                elif PARAMS["use_trail"] and trail_points:
                    trail_stop = state["highest_since_entry"] - trail_points
                    if low[i] <= trail_stop:
                        state["daily_pnl"] += (trail_stop - state["entry_price"]) * NQ_POINT_VALUE
                        exit_triggered = True
                
                if at_close:
                    state["daily_pnl"] += (close[i] - state["entry_price"]) * NQ_POINT_VALUE
                    exit_triggered = True
            
            elif state["position"] == -1:
                state["lowest_since_entry"] = min(state["lowest_since_entry"], low[i])
                
                if high[i] >= state["entry_price"] + sl_points:
                    state["daily_pnl"] -= sl_points * NQ_POINT_VALUE
                    exit_triggered = True
                elif low[i] <= state["entry_price"] - tp_points:
                    state["daily_pnl"] += tp_points * NQ_POINT_VALUE
                    exit_triggered = True
                elif PARAMS["use_trail"] and trail_points:
                    trail_stop = state["lowest_since_entry"] + trail_points
                    if high[i] >= trail_stop:
                        state["daily_pnl"] += (state["entry_price"] - trail_stop) * NQ_POINT_VALUE
                        exit_triggered = True
                
                if at_close:
                    state["daily_pnl"] += (state["entry_price"] - close[i]) * NQ_POINT_VALUE
                    exit_triggered = True
            
            if exit_triggered:
                state["position"] = 0
                signals[i] = 0
            else:
                signals[i] = state["position"]
        
        else:
            signals[i] = 0
    
    return signals
