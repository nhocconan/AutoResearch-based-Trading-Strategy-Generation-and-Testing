#!/usr/bin/env python3
"""
Experiment #021: 12h Camarilla + Choppiness Regime + Volume + 1d ATR Trend

HYPOTHESIS: 
- Previous attempt (#018) failed (Sharpe=-0.338) because choppiness was a soft filter.
- This version uses CHOPPINESS as a HARD regime filter:
  * CHOP > 61.8 (range) → only take LONG entries at S3/S4
  * CHOP < 38.2 (trending) → only take SHORT entries at R3/R4
- This mirrors the proven winning pattern: range = long mean-reversion, trend = short rallies.

WHY 12h: Slower than 4h = fewer trades = less fee drag.
Camarilla on 12h captures multi-day institutional levels.

TARGET: 75-150 total over 4 years. Size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_chop_regime_v2"
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
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (good for mean reversion)
    CHOP < 38.2 = trending (good for trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]) if idx > 0 else high[idx] - low[idx])
            sum_tr += tr
        
        highest_high = max(high[i-period+1:i+1])
        lowest_low = min(low[i-period+1:i+1])
        range_val = highest_high - lowest_low
        
        if range_val > 1e-10:
            chop[i] = 100 * (np.log(sum_tr) / np.log(range_val * period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR(14) for trend strength
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # 1d ATR ratio (short/long) for regime
    atr_1d_short = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=5)
    atr_1d_long = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=30)
    atr_ratio_1d = atr_1d_short / np.where(atr_1d_long > 0, atr_1d_long, 1)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Choppiness on 12h (regime filter)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop_12h[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME FILTER (1d ATR ratio) ===
        atr_ratio = atr_ratio_aligned[i] if not np.isnan(atr_ratio_aligned[i]) else 1.0
        regime_volatile = atr_ratio > 1.5  # High volatility regime
        
        # === CHOPPINESS REGIME (hard filter) ===
        chop = chop_12h[i]
        is_range_regime = chop > 61.8  # Ranging = good for long mean-reversion
        is_trend_regime = chop < 38.2   # Trending = good for short rallies
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA LEVELS from previous CLOSED bar ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Range regime + price touches S3/S4 + volume ===
            if is_range_regime and vol_spike:
                if low[i] <= s4:
                    desired_signal = SIZE
                elif low[i] <= s3:
                    desired_signal = SIZE
            
            # === SHORT: Trend regime + price touches R3/R4 + volume ===
            if is_trend_regime and vol_spike and regime_volatile:
                if high[i] >= r4:
                    desired_signal = -SIZE
                elif high[i] >= r3:
                    desired_signal = -SIZE
        
        # === ATR TRAILING STOP ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals