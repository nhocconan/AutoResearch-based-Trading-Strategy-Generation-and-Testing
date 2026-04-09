#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d timeframe + volume confirmation
# - Primary signal: Fade at R3/S3 levels, breakout continuation at R4/S4 levels from 1d Camarilla pivots
# - Volume filter: 6h volume > 20-period median volume to ensure participation
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla levels adapt to volatility, providing structure in all regimes

name = "6h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R4, R3, S3, S4
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to primary timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (fade level broken) OR above R4 (take profit at extreme)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (fade level broken) OR below S4 (take profit at extreme)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for fading opportunities at R3/S3 with volume confirmation
            # Fade long at S3: price touches/bounces off S3 level
            if (low[i] <= s3_aligned[i] * 1.001 and  # Allow small tolerance for touch
                close[i] > s3_aligned[i] and         # Close above S3 confirms bounce
                volume_regime[i]):
                position = 1
                signals[i] = 0.25
            # Fade short at R3: price touches/rejects R3 level
            elif (high[i] >= r3_aligned[i] * 0.999 and  # Allow small tolerance for touch
                  close[i] < r3_aligned[i] and         # Close below R3 confirms rejection
                  volume_regime[i]):
                position = -1
                signals[i] = -0.25
            # Breakout continuation: price breaks R4/S4 with volume
            elif (close[i] > r4_aligned[i] and volume_regime[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] < s4_aligned[i] and volume_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals