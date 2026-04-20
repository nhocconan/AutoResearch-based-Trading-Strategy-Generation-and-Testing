#!/usr/bin/env python3
# 4h_Donchian_20_VolumeSpike_Momentum
# Hypothesis: Trade Donchian channel (20-period) breakouts on 4h timeframe with volume spike confirmation and momentum filter.
# Enters long when price breaks above upper band with volume surge and positive momentum.
# Enters short when price breaks below lower band with volume surge and negative momentum.
# Uses 1d timeframe for Donchian calculation to reduce noise and improve reliability.
# Designed for 20-50 trades per year by requiring strong breakouts with volume confirmation.

name = "4h_Donchian_20_VolumeSpike_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper and lower bands
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Momentum filter: 10-period ROC (Rate of Change)
    roc = np.zeros_like(close)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper band with volume surge and positive momentum
            if (close[i] > upper_aligned[i] and 
                volume[i] > 2.0 * volume_ma[i] and 
                roc[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume surge and negative momentum
            elif (close[i] < lower_aligned[i] and 
                  volume[i] > 2.0 * volume_ma[i] and 
                  roc[i] < -0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower band or momentum turns negative
            if close[i] < lower_aligned[i] or roc[i] < -0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band or momentum turns positive
            if close[i] > upper_aligned[i] or roc[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals