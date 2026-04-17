#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d Volume Spike Regime.
Long when Williams %R < -80 (oversold) and 1d volume > 1.5 * 20-period average volume.
Short when Williams %R > -20 (overbought) and 1d volume > 1.5 * 20-period average volume.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Uses 1d for volume regime filter, 6h for Williams %R oscillator.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike filter: volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 6h Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        # Volume regime filter
        vol_regime = volume_spike_aligned[i] > 0.5  # True if volume spike
        
        if position == 0:
            # Long: Oversold AND volume spike regime
            if oversold and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND volume spike regime
            elif overbought and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR volume regime ends
            if exit_long or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR volume regime ends
            if exit_short or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dVolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0