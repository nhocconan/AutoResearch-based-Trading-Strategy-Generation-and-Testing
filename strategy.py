#!/usr/bin/env python3
name = "6h_1D_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data (HTF) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1D weekly trend filter: 5-day SMA on daily closes
    # Weekly trend = price above/below 5-day SMA (approx weekly trend)
    close_1d = df_1d['close'].values
    sma_5_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    sma_5_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_5_1d)
    
    # 6h Donchian channel (20-period)
    # Highest high and lowest low over last 20 periods
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(sma_5_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Break above Donchian upper + weekly uptrend + volume surge
            if (close[i] > high_20[i] and 
                close[i] > sma_5_1d_aligned[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower + weekly downtrend + volume surge
            elif (close[i] < low_20[i] and 
                  close[i] < sma_5_1d_aligned[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian band or trend reversal
            if position == 1:
                if (close[i] < low_20[i]) or (close[i] < sma_5_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > high_20[i]) or (close[i] > sma_5_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals