#!/usr/bin/env python3
"""
Experiment #229: 15m Opening Range Breakout + 1h/1d Trend Filter + RSI Confirmation

Hypothesis: 15m strategies failed (#217, #219, #221, #225) because entry conditions were 
too strict (0 trades = Sharpe=0.000). This version uses a PROVEN intraday pattern:

1. Opening Range (OR): First 1h of UTC day (4 bars @ 15m) defines high/low
2. OR Breakout: Long when price breaks OR high, Short when breaks OR low
3. 1h HMA Filter: Only trade in direction of 1h trend (reduces false breakouts)
4. 1d Bias: Prefer trades aligned with daily trend (HTF confluence)
5. RSI(7) Confirmation: RSI > 50 for longs, < 50 for shorts (not extreme values)

Why this should work on 15m:
- Opening Range breakouts are institutional patterns (high probability)
- 1h trend filter reduces whipsaw (HTF direction)
- RSI(7) > 50 is EASY to hit (not extreme like RSI < 30)
- Size = 0.20 (smaller for 15m frequency to reduce fee drag)
- Target: 50-100 trades/year (strict enough to avoid fee death)

Position Sizing: 0.20 base, 0.25 strong signals
Stoploss: 2.0x ATR trailing (tighter for lower TF)
Session: Prefer 00-12 UTC but allow all hours for trade frequency
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_or_breakout_rsi_1h1d_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

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
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20  # 20% base position size (smaller for 15m frequency)
    SIZE_STRONG = 0.25  # 25% for strong signals
    
    # Opening Range tracking (first 4 bars = 1 hour of UTC day)
    or_high = np.zeros(n)
    or_low = np.zeros(n)
    or_high[:] = np.nan
    or_low[:] = np.nan
    
    # Track day boundaries for OR calculation
    prev_day = -1
    day_start_idx = 0
    day_or_high = 0.0
    day_or_low = float('inf')
    
    for i in range(n):
        # Extract day from open_time (milliseconds timestamp)
        current_day = int(open_time[i] // (24 * 60 * 60 * 1000))
        
        if current_day != prev_day:
            # New day - reset OR tracking
            prev_day = current_day
            day_start_idx = i
            day_or_high = high[i]
            day_or_low = low[i]
        
        # Update OR high/low for first 4 bars (1 hour)
        bars_into_day = i - day_start_idx
        if bars_into_day < 4:
            day_or_high = max(day_or_high, high[i])
            day_or_low = min(day_or_low, low[i])
        
        # Store OR values for all bars after OR is complete
        if bars_into_day >= 4:
            or_high[i] = day_or_high
            or_low[i] = day_or_low
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Session hours (UTC) - 15m bars, so bar index within day
    # 00:00-12:00 UTC = bars 0-47 (48 bars * 15min = 12 hours)
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(or_high[i]) or np.isnan(or_low[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === OPENING RANGE BREAKOUT ===
        or_breakout_long = close[i] > or_high[i]
        or_breakout_short = close[i] < or_low[i]
        
        # === RSI CONFIRMATION (not extreme, just directional) ===
        rsi_bull = rsi_7[i] > 50.0
        rsi_bear = rsi_7[i] < 50.0
        
        # === SMA FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === SESSION FILTER (prefer 00-12 UTC but allow all) ===
        bars_into_day = i - day_start_idx if 'day_start_idx' in dir() else 0
        is_prime_session = bars_into_day < 48  # 00:00-12:00 UTC
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG SETUP: OR breakout + 1h bull + RSI bull
        if or_breakout_long and htf_1h_bull and rsi_bull:
            # Strong: also aligned with 1d trend
            if htf_1d_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT SETUP: OR breakout + 1h bear + RSI bear
        elif or_breakout_short and htf_1h_bear and rsi_bear:
            # Strong: also aligned with 1d trend
            if htf_1d_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x for 15m) ===
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