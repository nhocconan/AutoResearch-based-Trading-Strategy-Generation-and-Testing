#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Long when price > Alligator's Jaw (TEMA13) with 1w EMA50 uptrend and volume > 1.5x average
# Short when price < Alligator's Jaw with 1w EMA50 downtrend and volume > 1.5x average
# Exit when price crosses back to Alligator's Teeth (TEMA8)
# Williams Alligator uses smoothed moving averages (SMMA) to identify trends
# Williams Alligator: Jaw=TEMA13, Teeth=TEMA8, Lips=TEMA5
# Designed to capture trends with low frequency suitable for 12h timeframe
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_WilliamsAlligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=np.float64)
    result = np.empty_like(data, dtype=np.float64)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_VALUE) / N
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (using close prices)
    # Jaw: SMMA(13), Teeth: SMMA(8), Lips: SMMA(5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for SMMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Jaw, EMA50 uptrend, volume confirmation
            if (close[i] > jaw[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Jaw, EMA50 downtrend, volume confirmation
            elif (close[i] < jaw[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back to Teeth (or below)
            if close[i] <= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back to Teeth (or above)
            if close[i] >= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals