#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day high and weekly trend is up (close > 50-week SMA)
# Short when price breaks below 20-day low and weekly trend is down (close < 50-week SMA)
# Volume confirmation requires > 1.5x 20-day median volume
# Designed for low-frequency, high-conviction trades to minimize fee drag in both bull and bear markets
# Uses discrete position sizing (0.25) to limit turnover

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean()
    
    # Align all indicators to 1d timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20.values)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20.values)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w.values)
    
    # Volume confirmation: current > 1.5x 20-day median volume
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median_20
    vol_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold.values)
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_threshold_aligned[i])):
            continue
        
        # Long: price breaks above 20-day high, weekly trend up, volume spike
        if (close_1d[i] > high_20_aligned[i] and 
            close_1d[i] > sma_50_1w_aligned[i] and 
            volume_1d[i] > vol_threshold_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below 20-day low, weekly trend down, volume spike
        elif (close_1d[i] < low_20_aligned[i] and 
              close_1d[i] < sma_50_1w_aligned[i] and 
              volume_1d[i] > vol_threshold_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel or weekly trend reverses
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close_1d[i] < (high_20_aligned[i] + low_20_aligned[i]) / 2 or
                                          close_1d[i] < sma_50_1w_aligned[i])) or
               (signals[i-1] == -0.25 and (close_1d[i] > (high_20_aligned[i] + low_20_aligned[i]) / 2 or
                                           close_1d[i] > sma_50_1w_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0