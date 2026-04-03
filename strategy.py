#!/usr/bin/env python3
"""
Experiment #283: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation

HYPOTHESIS: 4h Donchian breakouts aligned with 12h HMA trend capture strong momentum 
with institutional participation. Volume confirmation (1.8x average) ensures follow-through. 
Designed for 4h timeframe to target 19-50 trades/year (75-200 over 4 years). 
Works in both bull and bear markets by only taking breakouts in direction of 12h HMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 12 // 2  # half of 12
        sqrt_len = int(np.sqrt(21))  # sqrt(21) ≈ 4
        
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        wma_2x_sub = 2 * wma_half - wma_full
        hma_12h = wma(wma_2x_sub, sqrt_len)
        
        # Pad to match original length
        hma_12h_padded = np.full(len(close_12h), np.nan)
        hma_12h_padded[half_len + sqrt_len - 1:half_len + sqrt_len + len(hma_12h)] = hma_12h
        
        # Align to 4h timeframe
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.8 if vol_ma_20[i] > 1e-10 else False  # 1.8x volume spike
        
        # --- 12h HMA Trend ---
        hma_trend_up = close[i] > hma_12h_aligned[i]
        hma_trend_down = close[i] < hma_12h_aligned[i]
        
        # --- Entry Conditions ---
        long_entry = bullish_breakout and vol_ok and hma_trend_up
        short_entry = bearish_breakout and vol_ok and hma_trend_down
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR 12h HMA trend turns down
                    if close[i] <= dc_lower_20[i] or not hma_trend_up:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR 12h HMA trend turns up
                    if close[i] >= dc_upper_20[i] or not hma_trend_down:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if long_entry:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>