#!/usr/bin/env python3
"""
Experiment #254: 1h Donchian Breakout with 4h/1d Trend Filter and Volume Confirmation
HYPOTHESIS: Use 4h/1d HTF for trend direction (price above/below EMA50) and 1h for precise Donchian(20) breakout entries.
Only take longs in uptrend when price breaks above 20-period high with volume > 1.5x average.
Only take shorts in downtrend when price breaks below 20-period low with volume > 1.5x average.
ATR(14) stoploss (2.5x) and discrete position sizing (0.20) to limit fee drag.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Session filter (08-20 UTC) avoids low-volume Asian session noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_254_1h_donchian_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for EMA50 trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === HTF: 1d data for EMA50 trend filter (stronger filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # Default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Enough for EMA50 and Donchian20
    
    for i in range(warmup, n):
        # Skip outside session (08-20 UTC) to reduce noise trades
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Trend Filter: Require BOTH 4h and 1d EMA50 agreement ---
        # Uptrend: price > EMA50 on both timeframes
        uptrend = (price > ema_50_4h_aligned[i]) and (price > ema_50_1d_aligned[i])
        # Downtrend: price < EMA50 on both timeframes
        downtrend = (price < ema_50_4h_aligned[i]) and (price < ema_50_1d_aligned[i])
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Signals ---
        # Long: price breaks above 20-period high
        long_breakout = price > highest_20[i]
        # Short: price breaks below 20-period low
        short_breakout = price < lowest_20[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position for minimum 2 bars to avoid whipsaw
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if volume_spike:
            # Uptrend + breakout high = long
            if uptrend and long_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Downtrend + breakout low = short
            elif downtrend and short_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals