#!/usr/bin/env python3
"""
Experiment #021: 12h Camarilla S4/R4 + Choppiness + Volume Regime

HYPOTHESIS: Camarilla S4/R4 are powerful reversal levels (deeper pivots used by
institutional traders). By filtering for choppy markets (CHOP > 50) and requiring
volume confirmation, this catches reversals at key levels with clean risk management.

WHY 12h: Slower timeframe = fewer trades = less fee drag. 12h captures multi-day swings.
S4/R4 are the "deep" Camarilla levels (10% of range from close) — better risk/reward than S3/R3.

WHY CHOPPINESS FILTER: If market is range-bound (CHOP > 50), Camarilla reversals fail more often.
Only trade when market has tendency to trend (CHOP < 50) OR during clear breakouts.

TARGET: 75-125 total trades over 4 years = 19-31/year. HARD MAX: 200.
Signal size: 0.25 (conservative — BTC crashed 77% in 2022).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_s4_chop_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — values > 50 = choppy/range, < 50 = trending"""
    n = len(close)
    chop = np.full(n, 100.0)
    
    for i in range(period, n):
        highest = np.max(high[i - period:i + 1])
        lowest = np.min(low[i - period:i + 1])
        range_sum = np.sum(high[i - period:i + 1] - low[i - period:i + 1])
        
        if highest - lowest > 1e-10 and range_sum > 1e-10:
            chop[i] = 100.0 * np.log10(range_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_50d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # === Local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Signals array
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FILTER: Skip if market is too choppy ===
        # CHOP > 61.8 = very choppy, CHOP < 38.2 = very trending
        # We want CHOP < 50 for Camarilla reversals to work
        if chop[i] >= 50.0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation (ratio > 1.5)
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        vol_spike = vol_ratio > 1.5
        
        # === CAMARILLA LEVELS from previous CLOSED bar ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla S4/R4 (factor 1.1/6 = 0.18333)
        r4 = prev_close + prev_range * 0.18333
        r3 = prev_close + prev_range * 0.09167
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price touches S4 with volume + trend alignment ===
            # Only S4 (deeper level = better risk/reward)
            if price_above_1d_ema and vol_spike and low[i] <= s4:
                desired_signal = SIZE
            
            # === SHORT: Price touches R4 with volume + trend alignment ===
            if not price_above_1d_ema and vol_spike and high[i] >= r4:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        # Update highest/lowest
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                # Stopped out?
                if low[i] < stop_price:
                    in_position = False
                    position_side = 0
                    desired_signal = 0.0
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                # Stopped out?
                if high[i] > stop_price:
                    in_position = False
                    position_side = 0
                    desired_signal = 0.0
        
        # === NEW ENTRY ===
        if in_position == False and desired_signal != 0.0:
            in_position = True
            position_side = int(np.sign(desired_signal))
            entry_price = close[i]
            entry_atr = atr_14[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            entry_bar = i
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
            else:
                stop_price = entry_price + 2.5 * entry_atr
        
        signals[i] = desired_signal
    
    return signals