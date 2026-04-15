#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 12-hour Exponential Moving Average crossover with volume confirmation
# EMA(12) on 1-day data provides trend direction; crossovers signal momentum shifts.
# Volume > 2.0x 50-day median ensures institutional participation.
# Designed to work in both bull (bullish crossovers) and bear (bearish crossovers) markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.
# Uses 1-day EMA on 1-week higher timeframe for trend filter to avoid whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA(12) on 1d close
    ema_12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate EMA(26) on 1d close for crossover
    ema_26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Volume filter: 2.0x 50-day median volume
    vol_median_50 = pd.Series(volume_1d).rolling(window=50, min_periods=50).median()
    vol_threshold = 2.0 * vol_median_50
    
    # Align 1d indicators to lower timeframe
    ema_12_aligned = align_htf_to_ltf(prices, df_1d, ema_12)
    ema_26_aligned = align_htf_to_ltf(prices, df_1d, ema_26)
    vol_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    signals = np.zeros(n)
    
    for i in range(1, n):  # Start from 1 to access previous values
        # Skip if any required data is NaN
        if (np.isnan(ema_12_aligned[i]) or np.isnan(ema_26_aligned[i]) or
            np.isnan(vol_threshold_aligned[i])):
            continue
        
        # Bullish crossover: EMA12 crosses above EMA26 + volume confirmation
        if (ema_12_aligned[i] > ema_26_aligned[i] and 
            ema_12_aligned[i-1] <= ema_26_aligned[i-1] and
            volume_1d[min(i, len(volume_1d)-1)] > vol_threshold_aligned[i]):
            signals[i] = 0.25
        
        # Bearish crossover: EMA12 crosses below EMA26 + volume confirmation
        elif (ema_12_aligned[i] < ema_26_aligned[i] and 
              ema_12_aligned[i-1] >= ema_26_aligned[i-1] and
              volume_1d[min(i, len(volume_1d)-1)] > vol_threshold_aligned[i]):
            signals[i] = -0.25
        
        # Exit: opposite crossover
        elif (ema_12_aligned[i] < ema_26_aligned[i] and 
              ema_12_aligned[i-1] >= ema_26_aligned[i-1] and
              signals[i-1] == 0.25):
            signals[i] = 0.0
        elif (ema_12_aligned[i] > ema_26_aligned[i] and 
              ema_12_aligned[i-1] <= ema_26_aligned[i-1] and
              signals[i-1] == -0.25):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_EMACrossover_VolumeFilter"
timeframe = "1d"
leverage = 1.0