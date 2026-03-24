#!/usr/bin/env python3
"""
Experiment #241: 15m Primary + 1h/4h/1d HTF — Intraday Mean Reversion with Daily Pivots

Hypothesis: 15m timeframe is underexplored (0 successful experiments). Key insight:
15m has too much noise for pure trend-following, but excellent for intraday mean
reversion WITH HTF trend filter. This strategy combines:

1. Daily Pivot Levels (from 1d HTF): Calculate previous day's H/L/C to get pivot points
   - Central Pivot (P) = (H + L + C) / 3
   - R1 = 2P - L, S1 = 2P - H
   - Long near S1 when 4h trend bull, Short near R1 when 4h trend bear

2. 4h HMA(21) for intermediate trend direction (must align with trade)

3. 15m RSI(7) for entry timing (oversold < 25 for long, overbought > 75 for short)

4. Session Filter: Only trade 00-12 UTC (London+NY overlap = highest crypto volume)

5. 1d HMA(50) for major trend bias (reduce size against major trend)

Position sizing: 0.15 base, 0.20 strong (smaller for 15m frequency)
Target: 50-100 trades/year, Sharpe > 0.40, DD > -35%

CRITICAL: Entry conditions loosened to ensure trades (RSI 25/75 not 10/90,
pivot proximity 1.5% not 0.5%, session filter 12h not 8h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pivot_rsi_session_1h4h1d_v1"
timeframe = "15m"
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
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_daily_pivots(df_1d):
    """
    Calculate daily pivot levels from 1d data
    Returns arrays aligned to 1d bars: pivot, r1, s1, r2, s2
    """
    n = len(df_1d)
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
    
    # Need at least 2 days to calculate pivots (use previous day's HLC)
    for i in range(1, n):
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        r1[i] = 2.0 * pivot[i] - prev_low
        s1[i] = 2.0 * pivot[i] - prev_high
        r2[i] = pivot[i] + (prev_high - prev_low)
        s2[i] = pivot[i] - (prev_high - prev_low)
    
    return pivot, r1, s1, r2, s2

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000.0
    utc_hour = (ts_seconds % 86400) / 3600.0
    return int(utc_hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate daily pivot levels from 1d data
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d = calculate_daily_pivots(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_15m = calculate_hma(close, period=21)
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15  # 15% base position size (smaller for 15m frequency)
    SIZE_STRONG = 0.20  # 20% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London+NY overlap) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 0 and utc_hour <= 12)
        
        # === HTF TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 15m TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === PIVOT PROXIMITY ===
        # Calculate distance to pivot levels as percentage
        pivot_dist = abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] if pivot_aligned[i] > 0 else 1.0
        s1_dist = abs(close[i] - s1_aligned[i]) / s1_aligned[i] if s1_aligned[i] > 0 else 1.0
        r1_dist = abs(close[i] - r1_aligned[i]) / r1_aligned[i] if r1_aligned[i] > 0 else 1.0
        
        near_s1 = s1_dist < 0.015  # Within 1.5% of S1
        near_r1 = r1_dist < 0.015  # Within 1.5% of R1
        near_pivot = pivot_dist < 0.01  # Within 1% of central pivot
        
        # === RSI CONDITIONS (loosened for trade generation) ===
        rsi_oversold = rsi_7[i] < 25.0  # Not too extreme (was < 10)
        rsi_overbought = rsi_7[i] > 75.0  # Not too extreme (was > 90)
        rsi_neutral = rsi_7[i] >= 35.0 and rsi_7[i] <= 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG SETUP: RSI oversold + near support + 4h trend bull
        if in_session and rsi_oversold:
            # Strong long: near S1 + 4h bull + 1d bull
            if near_s1 and htf_4h_bull and htf_1d_bull:
                desired_signal = SIZE_STRONG
            # Base long: near S1 or pivot + 4h bull
            elif (near_s1 or near_pivot) and htf_4h_bull:
                desired_signal = SIZE_BASE
            # Weaker long: RSI very oversold + 4h bull (pivot not required)
            elif rsi_7[i] < 20.0 and htf_4h_bull:
                desired_signal = SIZE_BASE
        
        # SHORT SETUP: RSI overbought + near resistance + 4h trend bear
        elif in_session and rsi_overbought:
            # Strong short: near R1 + 4h bear + 1d bear
            if near_r1 and htf_4h_bear and htf_1d_bear:
                desired_signal = -SIZE_STRONG
            # Base short: near R1 or pivot + 4h bear
            elif (near_r1 or near_pivot) and htf_4h_bear:
                desired_signal = -SIZE_BASE
            # Weaker short: RSI very overbought + 4h bear
            elif rsi_7[i] > 80.0 and htf_4h_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
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