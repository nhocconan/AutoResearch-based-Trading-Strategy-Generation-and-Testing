#!/usr/bin/env python3
"""
Experiment #011: 6h Donchian(20) Breakout + 1d Pivot Direction + Volume Confirmation

HYPOTHESIS: Combining a 6h Donchian breakout with 1d pivot level direction (using weekly pivot calculation from 1d data) 
and volume confirmation creates a strategy that trades with the higher timeframe structure while avoiding false breakouts. 
The weekly pivot provides institutional reference points, and trading only in alignment with the 1d pivot bias reduces 
whipsaws. Designed for 6h timeframe to balance trade frequency (~12-37/year) and capture medium-term swings in both 
bull and bear markets with controlled risk.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_1d_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_weekly_pivot_from_1d(df_1d):
    """
    Calculate weekly pivot points from daily OHLC data.
    Uses the prior week's high, low, close to calculate pivot levels for current week.
    """
    n = len(df_1d)
    if n < 5:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Convert to pandas Series for rolling operations
    high_series = pd.Series(df_1d['high'].values)
    low_series = pd.Series(df_1d['low'].values)
    close_series = pd.Series(df_1d['close'].values)
    
    # Calculate weekly high, low, close (prior week's values)
    weekly_high = high_series.rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = low_series.rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = close_series.rolling(window=5, min_periods=5).last().shift(1).values
    
    # Calculate pivot point and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot levels from 1d data
    pivot_1d, r1_1d, r2_1d, r3_1d, s1_1d, s2_1d, s3_1d = calculate_weekly_pivot_from_1d(df_1d)
    
    # Align HTF pivot levels to 6h timeframe (with shift(1) for completed weeks only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r2_1d + 2 * (r2_1d - s2_1d))  # R4 = R3 + (R3-S3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s2_1d - 2 * (r2_1d - s2_1d))  # S4 = S3 - (R3-S3)
    
    # === 6h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital) - conservative for 6h
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Pivot Direction Bias (using 1d weekly pivot) ---
        # Bullish bias: price above weekly pivot
        # Bearish bias: price below weekly pivot
        pivot_bullish = close[i] > pivot_aligned[i]
        pivot_bearish = close[i] < pivot_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Pivot Level Specific Logic ---
        # Fade at extreme levels (R3/S3), breakout continuation at extreme extremes (R4/S4)
        near_r3 = abs(close[i] - r3_aligned[i]) / close[i] < 0.005  # Within 0.5% of R3
        near_s3 = abs(close[i] - s3_aligned[i]) / close[i] < 0.005  # Within 0.5% of S3
        break_r4 = close[i] > r4_aligned[i]  # Break above R4
        break_s4 = close[i] < s4_aligned[i]  # Break below S4
        
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
            
            # Trend reversal exit (only after minimum hold to avoid premature exits)
            min_hold = (i - entry_bar) >= 3
            if min_hold:
                if position_side > 0 and pivot_bearish:
                    stop_hit = True
                if position_side < 0 and pivot_bullish:
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
        # 1. Breakout above R4 (strong bullish continuation) OR
        # 2. Pullback to S3 with bullish bias AND volume confirmation
        long_condition = (break_r4 and pivot_bullish and vol_ok) or \
                         (near_s3 and pivot_bullish and bullish_breakout and vol_ok)
        
        # Short conditions:
        # 1. Breakdown below S4 (strong bearish continuation) OR
        # 2. Pullback to R3 with bearish bias AND volume confirmation
        short_condition = (break_s4 and pivot_bearish and vol_ok) or \
                          (near_r3 and pivot_bearish and bearish_breakout and vol_ok)
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals