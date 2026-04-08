#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_spike_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume spike confirmation.
# Long at S1 support with volume spike, short at R1 resistance with volume spike.
# Camarilla levels provide statistically significant support/resistance in ranging markets,
# while volume spikes confirm institutional participation. Works in both bull and bear markets
# by fading extremes at proven pivot levels with volume confirmation.
# Target: 15-30 trades/year with ~0.25 position size to minimize fee drag.

name = "12h_1d_camarilla_pivot_volume_spike_v1"
timeframe = "12h"
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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # R2 = C + ((H-L)*1.1/6)
    # R1 = C + ((H-L)*1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L)*1.1/12)
    # S2 = C - ((H-L)*1.1/6)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    
    H_1d = df_1d['high'].values
    L_1d = df_1d['low'].values
    C_1d = df_1d['close'].values
    
    # Calculate pivot levels
    R1 = C_1d + ((H_1d - L_1d) * 1.1 / 12)
    S1 = C_1d - ((H_1d - L_1d) * 1.1 / 12)
    
    # Align to 12h timeframe (wait for daily bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: volume > 2x 24-period average (2 days of 12h data)
    vol_period = 24
    vol_ma = np.zeros_like(volume)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    vol_ma[:vol_period-1] = vol_ma[vol_period-1]
    
    # Start from sufficient lookback
    start_idx = max(vol_period, 1) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price reaches S1 again or volume drops
            if close[i] <= S1_aligned[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price reaches R1 again or volume drops
            if close[i] >= R1_aligned[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at S1 support with volume spike
            if abs(close[i] - S1_aligned[i]) < 0.001 * close[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price at R1 resistance with volume spike
            elif abs(close[i] - R1_aligned[i]) < 0.001 * close[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals