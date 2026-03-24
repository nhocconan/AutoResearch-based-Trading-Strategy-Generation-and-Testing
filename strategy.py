#!/usr/bin/env python3
"""
Experiment #140: 6h Primary + 1d/1w HTF — Woodie Pivot Breakout + HMA Trend

Hypothesis: After 120+ failed experiments, 6h timeframe remains unexplored with pivot-based strategies.
Woodie Pivot Points weight the close more heavily (H+L+2C)/4, making them more responsive to recent price action.
Combined with 1w HMA for major trend bias and 1d Choppiness filter, this should:
- Generate 30-60 trades/year (6h sweet spot between 4h noise and 12h slowness)
- Work on BTC/ETH bear markets (pivot bounces work in ranges)
- Avoid overfitting (simple pivot math, no complex regime switching)

Key design choices:
- Timeframe: 6h (NEW - zero prior experiments with pivots)
- HTF: 1w HMA(50) for major trend, 1d Choppiness for regime filter
- Entry: Price breaks Woodie R1/S1 with HMA confirmation
- Filter: CHOP < 55 to avoid breakout failures in choppy markets
- Position size: 0.27 (27% of capital, discrete levels)
- Stoploss: 2.5x ATR trailing stop
- Loose RSI filter (35-65) to ensure trade generation on ALL symbols

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_woodie_pivot_hma_chop_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use CHOP < 55 as filter to avoid false breakouts
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_woodie_pivots(high, low, close, prev_close):
    """
    Woodie Pivot Points
    Pivot = (H + L + 2*C) / 4
    R1 = 2*Pivot - Low
    S1 = 2*Pivot - High
    R2 = Pivot + (High - Low)
    S2 = Pivot - (High - Low)
    """
    n = len(close)
    pivot = np.zeros(n)
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    r2 = np.zeros(n)
    s2 = np.zeros(n)
    
    pivot[:] = np.nan
    r1[:] = np.nan
    s1[:] = np.nan
    r2[:] = np.nan
    s2[:] = np.nan
    
    for i in range(1, n):
        p = (high[i] + low[i] + 2.0 * close[i]) / 4.0
        pivot[i] = p
        r1[i] = 2.0 * p - low[i]
        s1[i] = 2.0 * p - high[i]
        r2[i] = p + (high[i] - low[i])
        s2[i] = p - (high[i] - low[i])
    
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d Choppiness for regime filter
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    pivot, r1, s1, r2, s2 = calculate_woodie_pivots(high, low, close, close)
    
    signals = np.zeros(n)
    SIZE = 0.27  # 27% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS FILTER (1d aligned) ===
        # Only take breakouts when CHOP < 55 (trending environment)
        chop_ok = chop_1d_aligned[i] < 55.0
        
        # === WOODIE PIVOT BREAKOUT SIGNALS ===
        # Breakout = close crosses above R1 or below S1
        # Use previous bar's pivot levels to avoid look-ahead
        pivot_breakout_bull = close[i] > r1[i-1]
        pivot_breakout_bear = close[i] < s1[i-1]
        
        # Strong breakout = breaks R2/S2
        strong_breakout_bull = close[i] > r2[i-1]
        strong_breakout_bear = close[i] < s2[i-1]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === RSI CONFIRMATION (LOOSE to ensure trades) ===
        rsi_ok_long = rsi[i] > 35.0
        rsi_ok_short = rsi[i] < 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Pivot breakout + HMA bull + HTF bull + chop filter + RSI ok
        if pivot_breakout_bull and hma_bull and htf_bull and chop_ok and rsi_ok_long:
            desired_signal = SIZE
        
        # SHORT: Pivot breakout + HMA bear + HTF bear + chop filter + RSI ok
        elif pivot_breakout_bear and hma_bear and htf_bear and chop_ok and rsi_ok_short:
            desired_signal = -SIZE
        
        # FALLBACK: Strong breakout (ignore HTF if very strong move)
        elif strong_breakout_bull and hma_bull and chop_ok and rsi[i] > 40.0:
            desired_signal = SIZE * 0.7
        elif strong_breakout_bear and hma_bear and chop_ok and rsi[i] < 60.0:
            desired_signal = -SIZE * 0.7
        
        # FALLBACK 2: HTF aligned but no breakout (pivot bounce play)
        elif htf_bull and hma_bull and close[i] > pivot[i] and rsi[i] > 45.0 and rsi[i] < 70.0:
            desired_signal = SIZE * 0.5
        elif htf_bear and hma_bear and close[i] < pivot[i] and rsi[i] < 55.0 and rsi[i] > 30.0:
            desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals