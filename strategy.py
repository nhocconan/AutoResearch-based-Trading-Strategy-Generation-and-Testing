#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Hypothesis: 6h Williams %R with 1d trend filter and volume spike.
    Williams %R identifies overbought/oversold conditions.
    1d EMA50 provides trend direction to avoid counter-trend trades.
    Volume spike confirms momentum behind the move.
    Works in bull/bear by only taking trades in direction of 1d trend.
    Target: 50-150 total trades over 4 years.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period Williams %R on 6h data
    willr = np.full(n, np.nan)
    if n >= 14:
        for i in range(13, n):
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high != lowest_low:
                willr[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                willr[i] = -50
    
    # Calculate 50-period EMA on 1d for trend filter
    ema50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(df_1d)):
            ema50_1d[i] = (close_1d[i] * 2 + ema50_1d[i-1] * 48) / 50
    
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period volume moving average on 1d
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_ma_20_6h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(willr[i]) or
            np.isnan(ema50_6h[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20_6h[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_6h[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 1.5
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + above 1d EMA50 + volume spike
            if willr[i] < -80 and close[i] > ema50_6h[i] and volume_ratio > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) + below 1d EMA50 + volume spike
            elif willr[i] > -20 and close[i] < ema50_6h[i] and volume_ratio > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Williams %R returns to neutral (> -50) or trend change
            if willr[i] > -50 or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Williams %R returns to neutral (< -50) or trend change
            if willr[i] < -50 or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsR_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0