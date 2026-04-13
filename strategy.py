#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter and structure
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h.values)
    
    # Daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.3x 20-period average
        # Approximate 6h volume from daily volume (assuming 4x 6h periods per day)
        volume_6h_approx = volume[i]  # Current 6h bar volume
        volume_ma_20_6h = volume_ma_20_1d_aligned[i] / 4  # Approximate 20-period average for 6h
        volume_condition = volume_6h_approx > (volume_ma_20_6h * 1.3)
        
        # Trend filter: only long when price > 12h EMA20, short when price < 12h EMA20
        long_trend = close[i] > ema_20_12h_aligned[i]
        short_trend = close[i] < ema_20_12h_aligned[i]
        
        # Entry conditions: trend + volume confirmation
        if position == 0:
            if long_trend and volume_condition:
                position = 1
                signals[i] = position_size
            elif short_trend and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price closes below 12h EMA20 or volume drops
            if close[i] < ema_20_12h_aligned[i] * 0.998:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price closes above 12h EMA20 or volume drops
            if close[i] > ema_20_12h_aligned[i] * 1.002:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h1d_EMA20_Volume_Filter"
timeframe = "6h"
leverage = 1.0