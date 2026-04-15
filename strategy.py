#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Works in both bull and bear by using 1d EMA200 to filter direction
# Long when price breaks above Donchian(20) high and close > 1d EMA200
# Short when price breaks below Donchian(20) low and close < 1d EMA200
# Volume confirmation: volume > 1.5x 20-period average
# Designed for low trade frequency (target 20-40/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if not enough data for indicators
        if np.isnan(ema_200_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(volume_ma[i]):
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Long conditions: breakout above Donchian high AND above 1d EMA200 AND volume confirmation
        if close[i] > high_max[i] and close[i] > ema_200_aligned[i] and vol_confirm and position <= 0:
            position = 1
            signals[i] = position_size
        # Short conditions: breakout below Donchian low AND below 1d EMA200 AND volume confirmation
        elif close[i] < low_min[i] and close[i] < ema_200_aligned[i] and vol_confirm and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit conditions: reverse signal or loss of trend filter
        elif position == 1 and (close[i] < ema_200_aligned[i] or close[i] < low_min[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_200_aligned[i] or close[i] > high_max[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0