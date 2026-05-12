#!/usr/bin/env python3
# 1d_WilsonReversal_1wTrend
# Hypothesis: Buy near weekly support (Wilson swing low) in uptrend, sell near weekly resistance (Wilson swing high) in downtrend.
# Uses Wilson price swing points on weekly timeframe for structure, with 1d EMA50 trend filter and volume confirmation.
# Designed for very low frequency (<20 trades/year) to avoid fee drag. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).

name = "1d_WilsonReversal_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wilson_swing_points(high, low, close):
    """
    Calculate Wilson swing points: swing highs and swing lows.
    Returns arrays of swing high and swing low values (NaN elsewhere).
    """
    n = len(high)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    # Need at least 5 points to identify swings
    for i in range(2, n - 2):
        # Swing high: higher than 2 bars on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = high[i]
        # Swing low: lower than 2 bars on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = low[i]
    
    return swing_high, swing_low

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Wilson swing points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Wilson swing points on weekly data
    swing_high_1w, swing_low_1w = wilson_swing_points(high_1w, low_1w, close_1w)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    swing_high_1w_aligned = align_htf_to_ltf(prices, df_1w, swing_high_1w)
    swing_low_1w_aligned = align_htf_to_ltf(prices, df_1w, swing_low_1w)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1d)  # Using weekly index for alignment
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Get nearest weekly swing levels (forward-filled from aligned array)
        # Since Wilson points are sparse, we use the most recent valid value
        swing_high_val = swing_high_1w_aligned[i]
        swing_low_val = swing_low_1w_aligned[i]
        
        if position == 0:
            # LONG: Price near weekly swing low in uptrend with volume
            if (not np.isnan(swing_low_val) and 
                abs(high[i] - swing_low_val) / swing_low_val < 0.02 and  # Within 2% of swing low
                trend_up and vol_ok):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near weekly swing high in downtrend with volume
            elif (not np.isnan(swing_high_val) and 
                  abs(low[i] - swing_high_val) / swing_high_val < 0.02 and  # Within 2% of swing high
                  trend_down and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches weekly swing high or trend reverses
            if (not np.isnan(swing_high_val) and 
                abs(low[i] - swing_high_val) / swing_high_val < 0.02) or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly swing low or trend reverses
            if (not np.isnan(swing_low_val) and 
                abs(high[i] - swing_low_val) / swing_low_val < 0.02) or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals