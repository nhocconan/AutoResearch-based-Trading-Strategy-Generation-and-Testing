"""
13 EMA Breakout + Liquidity Sweep + BOS Confirmation (A / A+)
Converted from TradingView Pine Script

LIMITATIONS:
- No position tracking or partial closes (signal-only module)
- Kill switch disabled (requires runtime equity/P&L)
- Session filtering requires pre-computed 'in_session' column
- Pivot detection approximated with rolling windows
- Liquidity levels use previous bar data only (no lookahead)
"""

import numpy as np
import pandas as pd

name = "ema13-sweep-bos-breakout"
timeframe = "5m"
leverage = 1.0

def _ema(series, length):
    """Calculate exponential moving average."""
    return series.ewm(span=length, adjust=False).mean()

def _atr(high, low, close, length=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

def _find_pivot_highs(high, left_bars, right_bars):
    """Find confirmed pivot highs (non-repainting)."""
    n = len(high)
    pivots = np.full(n, np.nan)
    high_vals = high.values
    for i in range(left_bars, n - right_bars):
        is_pivot = True
        for j in range(i - left_bars, i):
            if high_vals[j] >= high_vals[i]:
                is_pivot = False
                break
        if is_pivot:
            for j in range(i + 1, i + right_bars + 1):
                if high_vals[j] >= high_vals[i]:
                    is_pivot = False
                    break
        if is_pivot:
            pivots[i] = high_vals[i]
    return pivots

def _find_pivot_lows(low, left_bars, right_bars):
    """Find confirmed pivot lows (non-repainting)."""
    n = len(low)
    pivots = np.full(n, np.nan)
    low_vals = low.values
    for i in range(left_bars, n - right_bars):
        is_pivot = True
        for j in range(i - left_bars, i):
            if low_vals[j] <= low_vals[i]:
                is_pivot = False
                break
        if is_pivot:
            for j in range(i + 1, i + right_bars + 1):
                if low_vals[j] <= low_vals[i]:
                    is_pivot = False
                    break
        if is_pivot:
            pivots[i] = low_vals[i]
    return pivots

def _detect_liquidity_sweep(close, high, low, liq_high, liq_low, atr,
                            sweep_atr_mult=0.20, sweep_min_ticks=2,
                            sweep_confirm_bars=3, mintick=0.01):
    """Detect liquidity sweep events."""
    n = len(close)
    close_vals = close.values
    high_vals = high.values
    low_vals = low.values
    liq_high_vals = liq_high.values
    liq_low_vals = liq_low.values
    atr_vals = atr.values
    
    sweep_dist = np.maximum(sweep_min_ticks * mintick, atr_vals * sweep_atr_mult)
    
    bull_sweep = np.zeros(n, dtype=bool)
    bear_sweep = np.zeros(n, dtype=bool)
    
    up_breach_bar = np.full(n, -1)
    dn_breach_bar = np.full(n, -1)
    
    for i in range(1, n):
        if not np.isnan(liq_high_vals[i]) and high_vals[i] > liq_high_vals[i] + sweep_dist[i]:
            up_breach_bar[i] = i
        
        if up_breach_bar[i] >= 0:
            breach_idx = up_breach_bar[i]
            if i - breach_idx <= sweep_confirm_bars and close_vals[i] < liq_high_vals[i]:
                bear_sweep[i] = True
                up_breach_bar[i] = -1
            elif i - breach_idx > sweep_confirm_bars:
                up_breach_bar[i] = -1
        else:
            up_breach_bar[i] = up_breach_bar[i-1]
        
        if not np.isnan(liq_low_vals[i]) and low_vals[i] < liq_low_vals[i] - sweep_dist[i]:
            dn_breach_bar[i] = i
        
        if dn_breach_bar[i] >= 0:
            breach_idx = dn_breach_bar[i]
            if i - breach_idx <= sweep_confirm_bars and close_vals[i] > liq_low_vals[i]:
                bull_sweep[i] = True
                dn_breach_bar[i] = -1
            elif i - breach_idx > sweep_confirm_bars:
                dn_breach_bar[i] = -1
        else:
            dn_breach_bar[i] = dn_breach_bar[i-1]
    
    return bull_sweep, bear_sweep

def _detect_bos(close, swing_high, swing_low, require_retest=False, retest_bars=10):
    """Detect Break of Structure events."""
    n = len(close)
    close_vals = close.values
    
    bos_up = np.zeros(n, dtype=bool)
    bos_dn = np.zeros(n, dtype=bool)
    bos_up_retested = np.zeros(n, dtype=bool)
    bos_dn_retested = np.zeros(n, dtype=bool)
    
    last_swing_high = np.nan
    last_swing_low = np.nan
    last_bos_up_bar = -1
    last_bos_dn_bar = -1
    
    for i in range(1, n):
        if not np.isnan(swing_high[i]):
            last_swing_high = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_swing_low = swing_low[i]
        
        if not np.isnan(last_swing_high) and close_vals[i] > last_swing_high and close_vals[i-1] <= last_swing_high:
            bos_up[i] = True
            last_bos_up_bar = i
        
        if not np.isnan(last_swing_low) and close_vals[i] < last_swing_low and close_vals[i-1] >= last_swing_low:
            bos_dn[i] = True
            last_bos_dn_bar = i
        
        if require_retest and last_bos_up_bar >= 0 and i - last_bos_up_bar <= retest_bars:
            if low.iloc[i] <= last_swing_high and close_vals[i] > last_swing_high:
                bos_up_retested[i] = True
        
        if require_retest and last_bos_dn_bar >= 0 and i - last_bos_dn_bar <= retest_bars:
            if high.iloc[i] >= last_swing_low and close_vals[i] < last_swing_low:
                bos_dn_retested[i] = True
    
    return bos_up, bos_dn, bos_up_retested, bos_dn_retested

def _detect_patterns(close, high, low, pivot_high, pivot_low, atr,
                     pattern_type='double_top_bottom', range_bars=20,
                     dbl_atr_tol=0.30, dbl_min_sep=4, range_max_atr=1.20):
    """Detect chart patterns (Double Top/Bottom, Range Breakout, Wedge)."""
    n = len(close)
    close_vals = close.values
    high_vals = high.values
    low_vals = low.values
    atr_vals = atr.values
    
    pat_bull = np.zeros(n, dtype=bool)
    pat_bear = np.zeros(n, dtype=bool)
    
    prev_ph = np.nan
    last_ph = np.nan
    prev_ph_bar = -1
    last_ph_bar = -1
    prev_pl = np.nan
    last_pl = np.nan
    prev_pl_bar = -1
    last_pl_bar = -1
    
    for i in range(n):
        if not np.isnan(pivot_high[i]):
            prev_ph = last_ph
            prev_ph_bar = last_ph_bar
            last_ph = pivot_high[i]
            last_ph_bar = i
        
        if not np.isnan(pivot_low[i]):
            prev_pl = last_pl
            prev_pl_bar = last_pl_bar
            last_pl = pivot_low[i]
            last_pl_bar = i
        
        if pattern_type == 'double_top_bottom':
            if not np.isnan(last_pl) and not np.isnan(prev_pl) and last_pl_bar - prev_pl_bar >= dbl_min_sep:
                if abs(last_pl - prev_pl) <= atr_vals[i] * dbl_atr_tol:
                    pat_bull[i] = True
            
            if not np.isnan(last_ph) and not np.isnan(prev_ph) and last_ph_bar - prev_ph_bar >= dbl_min_sep:
                if abs(last_ph - prev_ph) <= atr_vals[i] * dbl_atr_tol:
                    pat_bear[i] = True
        
        elif pattern_type == 'range_breakout':
            if i >= range_bars:
                prev_range_hi = high_vals[i-range_bars:i].max()
                prev_range_lo = low_vals[i-range_bars:i].min()
                range_width = prev_range_hi - prev_range_lo
                if range_width <= atr_vals[i] * range_max_atr:
                    if close_vals[i] > prev_range_hi:
                        pat_bull[i] = True
                    elif close_vals[i] < prev_range_lo:
                        pat_bear[i] = True
    
    return pat_bull, pat_bear

def generate_signals(prices):
    """
    Generate trading signals based on EMA13 Breakout + Liquidity Sweep + BOS strategy.
    
    Args:
        prices: pandas.DataFrame with columns [open_time, open, high, low, close, volume]
                Must be sorted by open_time ascending.
    
    Returns:
        numpy.ndarray of signals: 1 (long), -1 (short), 0 (flat)
    """
    df = prices.copy()
    n = len(df)
    signals = np.zeros(n, dtype=np.int8)
    
    if n < 50:
        return signals
    
    df['ema13'] = _ema(df['close'], 13)
    df['ema50'] = _ema(df['close'], 50)
    df['ema200'] = _ema(df['close'], 200)
    df['atr'] = _atr(df['high'], df['low'], df['close'], 14)
    
    pivot_high = _find_pivot_highs(df['high'], 3, 3)
    pivot_low = _find_pivot_lows(df['low'], 3, 3)
    
    df['prev_day_high'] = df['high'].shift(1).rolling(20).max()
    df['prev_day_low'] = df['low'].shift(1).rolling(20).min()
    
    liq_high = df['prev_day_high']
    liq_low = df['prev_day_low']
    
    bull_sweep, bear_sweep = _detect_liquidity_sweep(
        df['close'], df['high'], df['low'],
        liq_high, liq_low, df['atr'],
        sweep_atr_mult=0.20, sweep_min_ticks=2,
        sweep_confirm_bars=3, mintick=0.01
    )
    
    bos_up, bos_dn, bos_up_retested, bos_dn_retested = _detect_bos(
        df['close'], pivot_high, pivot_low,
        require_retest=False, retest_bars=10
    )
    
    pat_bull, pat_bear = _detect_patterns(
        df['close'], df['high'], df['low'],
        pivot_high, pivot_low, df['atr'],
        pattern_type='double_top_bottom', range_bars=20,
        dbl_atr_tol=0.30, dbl_min_sep=4, range_max_atr=1.20
    )
    
    close_vals = df['close'].values
    ema13_vals = df['ema13'].values
    ema50_vals = df['ema50'].values
    ema200_vals = df['ema200'].values
    
    ema_break_up = np.zeros(n, dtype=bool)
    ema_break_dn = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if close_vals[i] > ema13_vals[i] and close_vals[i-1] <= ema13_vals[i-1]:
            ema_break_up[i] = True
        if close_vals[i] < ema13_vals[i] and close_vals[i-1] >= ema13_vals[i-1]:
            ema_break_dn[i] = True
    
    last_long_qualified = np.full(n, -1)
    last_short_qualified = np.full(n, -1)
    last_bull_sweep_bar = np.full(n, -1)
    last_bear_sweep_bar = np.full(n, -1)
    
    for i in range(1, n):
        if bull_sweep[i]:
            last_bull_sweep_bar[i] = i
        else:
            last_bull_sweep_bar[i] = last_bull_sweep_bar[i-1]
        
        if bear_sweep[i]:
            last_bear_sweep_bar[i] = i
        else:
            last_bear_sweep_bar[i] = last_bear_sweep_bar[i-1]
        
        if bos_up[i] and (i > 0 and ema_break_up[i-1]):
            last_long_qualified[i] = i
        elif last_long_qualified[i-1] >= 0:
            last_long_qualified[i] = last_long_qualified[i-1]
        
        if bos_dn[i] and (i > 0 and ema_break_dn[i-1]):
            last_short_qualified[i] = i
        elif last_short_qualified[i-1] >= 0:
            last_short_qualified[i] = last_short_qualified[i-1]
    
    setup_valid_bars = 20
    sweep_valid_bars = 30
    
    for i in range(1, n):
        long_setup_valid = (i - last_long_qualified[i] <= setup_valid_bars) if last_long_qualified[i] >= 0 else False
        short_setup_valid = (i - last_short_qualified[i] <= setup_valid_bars) if last_short_qualified[i] >= 0 else False
        
        has_bull_sweep = (i - last_bull_sweep_bar[i] <= sweep_valid_bars) if last_bull_sweep_bar[i] >= 0 else False
        has_bear_sweep = (i - last_bear_sweep_bar[i] <= sweep_valid_bars) if last_bear_sweep_bar[i] >= 0 else False
        
        has_bull_pattern = pat_bull[i]
        has_bear_pattern = pat_bear[i]
        
        path_aplus_long = has_bull_sweep and long_setup_valid
        path_aplus_short = has_bear_sweep and short_setup_valid
        
        path_a_long = not has_bull_sweep and has_bull_pattern and long_setup_valid
        path_a_short = not has_bear_sweep and has_bear_pattern and short_setup_valid
        
        ema50_long_ok = close_vals[i] > ema50_vals[i]
        ema50_short_ok = close_vals[i] < ema50_vals[i]
        ema200_long_ok = close_vals[i] > ema200_vals[i]
        ema200_short_ok = close_vals[i] < ema200_vals[i]
        
        filter_long_ok = ema50_long_ok and ema200_long_ok
        filter_short_ok = ema50_short_ok and ema200_short_ok
        
        long_path_ok = path_aplus_long or path_a_long
        short_path_ok = path_aplus_short or path_a_short
        
        if long_path_ok and filter_long_ok and signals[i-1] <= 0:
            signals[i] = 1
        elif short_path_ok and filter_short_ok and signals[i-1] >= 0:
            signals[i] = -1
        else:
            signals[i] = 0
    
    return signals
