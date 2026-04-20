#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Volume Spike + Trend Filter
# - Williams %R (14) on 4h for momentum reversal signals
# - Long when %R < -80 (oversold) and 1d volume > 1.5x 20-period average (institutional interest)
# - Short when %R > -20 (overbought) and 1d volume > 1.5x 20-period average
# - Williams %R identifies overextended moves; volume spike confirms institutional participation
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume analysis
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d timeframe
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (1.5 * avg_vol_20)
    
    # Align 1d volume spike to 4h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate Williams %R (14) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Williams %R warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(vol_spike_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        wr = williams_r[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # Convert to boolean
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + volume spike
            if wr < -80 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + volume spike
            elif wr > -20 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or volume spike ends
            if wr > -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or volume spike ends
            if wr < -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_VolumeSpike"
timeframe = "4h"
leverage = 1.0