#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot with 1d volume regime filter
# - Long when price breaks above R3 and 1d volume is in expansion regime (vol > 1.5x 20-day avg)
# - Short when price breaks below S3 and 1d volume is in expansion regime
# - Exit when price returns to Camarilla pivot point (PP)
# - Volume regime filter ensures we only trade breakouts with institutional participation
# - Works in both bull (continuation breakouts) and bear (panic breakdowns) markets
# - Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag

name = "6h_1d_camarilla_pivot_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume regime: expansion when vol > 1.5x 20-day average
    volume_20d_avg = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_expansion = df_1d['volume'] > (1.5 * volume_20d_avg)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    # Pre-compute 12h data for Camarilla calculation (using 12h as midpoint between 6h and 1d)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    pp_12h = typical_price_12h  # PP = (H+L+C)/3
    range_12h = high_12h - low_12h
    r3_12h = pp_12h + (range_12h * 1.1 / 2.0)
    s3_12h = pp_12h - (range_12h * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (completed 12h bar only)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(pp_12h_aligned[i]) or np.isnan(vol_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price closes above R3 with 1d volume expansion
            if (prices['close'].iloc[i] > r3_12h_aligned[i] and 
                vol_expansion_aligned.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price closes below S3 with 1d volume expansion
            elif (prices['close'].iloc[i] < s3_12h_aligned[i] and 
                  vol_expansion_aligned.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to Camarilla pivot point (PP)
            if position == 1 and prices['close'].iloc[i] < pp_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > pp_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals