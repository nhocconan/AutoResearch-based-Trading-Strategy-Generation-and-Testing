#!/usr/bin/env python3
"""
Experiment #298: 1d Donchian(20) Breakout + Weekly Trend + Volume Confirmation

HYPOTHESIS: Daily Donchian(20) breakouts aligned with weekly trend (HMA crossover) and 
volume confirmation (1.5x average) capture sustained momentum with institutional participation. 
Weekly trend filter reduces false breakouts by ensuring alignment with higher timeframe momentum. 
Designed for 1d timeframe to target 15-25 trades/year (60-100 over 4 years), minimizing fee drag. 
Works in both bull and bear markets by only taking breakouts in direction of weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly trend calculation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA(21) for trend direction
    if len(df_1w) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = len(df_1w) // 2
        sqrt_len = int(np.sqrt(len(df_1w)))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights/weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values
        if len(close_1w) >= 21:
            wma_half = wma(close_1w, half_len) if half_len >= 1 else np.array([])
            wma_full = wma(close_1w, len(close_1w))
            if len(wma_half) > 0 and len(wma_full) > 0:
                raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
                if len(raw_hma) >= sqrt_len:
                    hma_values = wma(raw_hma, sqrt_len)
                    # Pad to match original length
                    hma_1w = np.full(len(close_1w), np.nan)
                    hma_1w[-len(hma_values):] = hma_values
                    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
                else:
                    hma_1w_aligned = np.full(n, np.nan)
            else:
                hma_1w_aligned = np.full(n, np.nan)
        else:
            hma_1w_aligned = np.full(n, np.nan)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
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
            np.isnan(vol_ma_20[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Weekly Trend Filter ---
        # Weekly trend: 1 if price > weekly HMA (bullish), -1 if price < weekly HMA (bearish)
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
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
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~48h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR weekly trend turns bearish
                    if close[i] <= dc_lower_20[i] or weekly_trend < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR weekly trend turns bullish
                    if close[i] >= dc_upper_20[i] or weekly_trend > 0:
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
        # Long conditions: 
        # Breakout above upper Donchian with volume confirmation, weekly trend bullish
        if bullish_breakout and vol_ok and weekly_trend > 0:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation, weekly trend bearish
        elif bearish_breakout and vol_ok and weekly_trend < 0:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals