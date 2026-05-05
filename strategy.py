#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) AND price > lips (5-period SMMA) 
# AND volume > 1.5x 20-period average AND close > 1d EMA50
# Short when jaws cross below teeth AND price < lips AND volume > 1.5x 20-period average AND close < 1d EMA50
# Exit when jaws re-cross teeth in opposite direction (Alligator "sleeping" signal)
# Uses 4h primary timeframe for execution with volume from same timeframe
# 1d HTF for EMA trend filter to avoid counter-trend trades
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Williams_Alligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple SMA
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA(i) = (SMMA(i-1) * (period-1) + source[i]) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume spike filter on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 4h timeframe
    # Jaws: 13-period SMMA of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2
    
    jaws = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: jaws crosses above teeth AND price > lips AND volume spike AND above 1d EMA50
            if (jaws[i] > teeth[i] and jaws[i-1] <= teeth[i-1] and  # jaws crossing above teeth
                close[i] > lips[i] and 
                volume_filter[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: jaws crosses below teeth AND price < lips AND volume spike AND below 1d EMA50
            elif (jaws[i] < teeth[i] and jaws[i-1] >= teeth[i-1] and  # jaws crossing below teeth
                  close[i] < lips[i] and 
                  volume_filter[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: jaws crosses below teeth (Alligator sleeping - trend weakening)
            if jaws[i] < teeth[i] and jaws[i-1] >= teeth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: jaws crosses above teeth (Alligator sleeping - trend weakening)
            if jaws[i] > teeth[i] and jaws[i-1] <= teeth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals