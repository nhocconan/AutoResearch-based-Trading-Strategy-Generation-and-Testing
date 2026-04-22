#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume confirmation.
Long when price breaks above upper Donchian band with bullish 1w trend and volume spike.
Short when price breaks below lower Donchian band with bearish 1w trend and volume spike.
Exit when price returns to middle Donchian band (20-day SMA).
Uses 1w EMA21 for trend filter to capture major trend and avoid whipsaws.
Designed for low trade frequency (7-25/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 22:
        return np.zeros(n)
    
    # Calculate 1w EMA21 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA21 to 1d timeframe
    ema21_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Calculate 20-period Donchian channels on 1d
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band is 20-day SMA of close
    mid_band = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period volume average for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after lookback period
        # Skip if data not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(mid_band[i]) or np.isnan(ema21_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian band with bullish 1w trend and volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema21_aligned[i] and  # Bullish trend: price above EMA21
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band with bearish 1w trend and volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema21_aligned[i] and  # Bearish trend: price below EMA21
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to middle Donchian band
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle band
                if close[i] <= mid_band[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle band
                if close[i] >= mid_band[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0
#%%