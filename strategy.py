#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels + volume confirmation
# Camarilla pivot levels from 12h provide strong support/resistance with proven edge on BTC/ETH
# Volume confirmation filters false breakouts (current 6h volume > 1.8x 20-period average)
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25
# Works in bull/bear: price reacts to pivot levels, volume confirms validity

name = "6h_12h_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    pivot = (high_12h + low_12h + close_12h) / 3.0
    rng = high_12h - low_12h
    
    # Camarilla levels: L3, L4, H3, H4
    camarilla_l3 = pivot - (1.1 * rng / 2)
    camarilla_l4 = pivot - (1.1 * rng)
    camarilla_h3 = pivot + (1.1 * rng / 2)
    camarilla_h4 = pivot + (1.1 * rng)
    
    # Align 12h Camarilla levels to 6h timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x average 6h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit when price reaches H3 level (take profit)
            if close[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches L3 level (take profit)
            if close[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion at Camarilla levels with volume confirmation
            # Long near L4, Short near H4
            if volume_confirmed:
                if close[i] <= camarilla_l4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= camarilla_h4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals