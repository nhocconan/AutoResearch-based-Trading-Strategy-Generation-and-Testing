#!/usr/bin/env python3
"""
Experiment #009: 15m Primary + 1h/1d HTF — Opening Range Breakout + Daily CPR + Session Filter

Hypothesis: 15m strategies failed before due to too many trades and weak confluence.
This strategy uses PROVEN intraday patterns with STRICT filters:
- Opening Range (first 1h of UTC day) breakout in direction of daily bias
- Daily CPR (Central Pivot Range) from 1d HTF for support/resistance
- 1h HMA for intermediate trend confirmation
- Session filter: only trade 00:00-12:00 UTC (London+NY overlap = highest volume)
- Choppiness Index < 55 to avoid range-bound conditions
- RSI(7) momentum confirmation (not extreme, just directional)
- Position size: 0.18 (conservative for 15m frequency)
- Stoploss: 2.0x ATR trailing

Why this might work on 15m:
- Opening Range breakouts have statistical edge across all timeframes
- Session filter reduces trades by ~60% (only 12h of 24h)
- Daily CPR provides concrete S/R levels from HTF
- 3+ confluence prevents overtrading (target 50-80 trades/year)

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_orb_cpr_session_1h1d_v1"
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
    We use threshold 55 for regime detection
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

def calculate_daily_cpr(df_1d):
    """
    Calculate Daily Central Pivot Range (CPR) from 1d data
    CPR = (Pivot, BC, TC) where:
    Pivot = (High + Low + Close) / 3
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = (Pivot - BC) + Pivot
    
    Returns arrays aligned to 1d bars
    """
    n = len(df_1d)
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot - bc) + pivot
    
    return pivot, bc, tc

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for intermediate trend
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 1d CPR for daily bias
    pivot_1d, bc_1d, tc_1d = calculate_daily_cpr(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    
    # Calculate 1d HMA for major trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Track opening range (first 4 bars of each UTC day = 1 hour)
    opening_range_high = np.zeros(n)
    opening_range_low = np.zeros(n)
    opening_range_high[:] = np.nan
    opening_range_low[:] = np.nan
    
    # Build opening range by tracking daily sessions
    current_day = -1
    day_or_high = 0.0
    day_or_low = float('inf')
    or_bars_counted = 0
    
    for i in range(n):
        utc_hour = get_utc_hour(open_time[i])
        day_id = int(open_time[i] // 86400000)  # Day identifier
        
        if day_id != current_day:
            # New day - reset OR tracking
            current_day = day_id
            day_or_high = high[i]
            day_or_low = low[i]
            or_bars_counted = 1
        elif or_bars_counted < 4:
            # Still within first hour (4 x 15m bars)
            day_or_high = max(day_or_high, high[i])
            day_or_low = min(day_or_low, low[i])
            or_bars_counted += 1
        
        # Store OR levels for all bars of the day
        if or_bars_counted >= 1:
            opening_range_high[i] = day_or_high
            opening_range_low[i] = day_or_low
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (conservative for 15m frequency)
    
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
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(opening_range_high[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00:00-12:00 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 0 and utc_hour < 12)
        
        if not in_session:
            # Close existing positions outside session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === CHOPPINESS FILTER (only trade trending conditions) ===
        is_trending = chop[i] < 55.0
        
        if not is_trending:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA + CPR) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > pivot_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < pivot_aligned[i]
        
        # === 1h INTERMEDIATE TREND ===
        h1_bull = close[i] > hma_1h_aligned[i]
        h1_bear = close[i] < hma_1h_aligned[i]
        
        # === OPENING RANGE BREAKOUT ===
        or_breakout_bull = close[i] > opening_range_high[i-1]  # Break above OR high
        or_breakout_bear = close[i] < opening_range_low[i-1]   # Break below OR low
        
        # === CPR LEVEL POSITION ===
        # Price above TC = bullish bias, below BC = bearish bias
        above_tc = close[i] > tc_aligned[i]
        below_bc = close[i] < bc_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI MOMENTUM (not extreme, just directional) ===
        rsi_bull = rsi[i] > 45.0 and rsi[i] < 70.0  # Bullish momentum, not overbought
        rsi_bear = rsi[i] < 55.0 and rsi[i] > 30.0  # Bearish momentum, not oversold
        
        # === DESIRED SIGNAL (Multiple Confluence) ===
        desired_signal = 0.0
        
        # LONG: OR breakout + HTF bull + 1h bull + RSI bull + above TC or neutral CPR
        long_confluence = (
            or_breakout_bull and
            (htf_bull or above_tc) and
            h1_bull and
            rsi_bull and
            hma_bull
        )
        
        # SHORT: OR breakout + HTF bear + 1h bear + RSI bear + below BC or neutral CPR
        short_confluence = (
            or_breakout_bear and
            (htf_bear or below_bc) and
            h1_bear and
            rsi_bear and
            hma_bear
        )
        
        if long_confluence:
            desired_signal = SIZE
        elif short_confluence:
            desired_signal = -SIZE
        # Reduced size for weaker confluence (4/5 factors)
        elif or_breakout_bull and h1_bull and rsi_bull and hma_bull:
            desired_signal = SIZE * 0.6
        elif or_breakout_bear and h1_bear and rsi_bear and hma_bear:
            desired_signal = -SIZE * 0.6
        
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