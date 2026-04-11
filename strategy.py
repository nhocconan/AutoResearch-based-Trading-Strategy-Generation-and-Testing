#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with volume confirmation and 1d trend filter
# Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) + volume > 1.5x average + 1d trend up
# Short when jaws crosses below teeth + volume > 1.5x average + 1d trend down
# Exit when price crosses back below jaws (long) or above jaws (short)
# Alligator uses smoothed moving averages for smoother trend signals
# Designed for 20-50 trades/year on 4h timeframe with strong trend capture and low turnover

name = "4h_1d_alligator_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Alligator components (13, 8, 5 period SMMA of median price)
    median_price = (high + low) / 2
    jaws = smma(median_price, 13)   # Blue line (13-period)
    teeth = smma(median_price, 8)   # Red line (8-period)
    lips = smma(median_price, 5)    # Green line (5-period)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after jaws period
        # Skip if any required data is invalid
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: Jaw crosses Teeth
        jaw_crosses_teeth_up = jaws[i] > teeth[i] and jaws[i-1] <= teeth[i-1]
        jaw_crosses_teeth_down = jaws[i] < teeth[i] and jaws[i-1] >= teeth[i-1]
        
        long_entry = jaw_crosses_teeth_up and volume_filter and is_uptrend
        short_entry = jaw_crosses_teeth_down and volume_filter and is_downtrend
        
        # Exit conditions: Price crosses back below/above Jaw
        long_exit = (position == 1 and close[i] < jaws[i])
        short_exit = (position == -1 and close[i] > jaws[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals