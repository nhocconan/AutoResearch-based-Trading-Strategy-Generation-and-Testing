#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla pivot levels with volume confirmation
# Camarilla pivots from 1w provide strong weekly structure aligned with 12h timeframe
# Volume confirmation (current 12h volume > 1.8x 20-period average) filters false breakouts
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# Works in bull/bear: price reacts to weekly structure, volume confirms validity
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "12h_1w_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # Resistance levels: R1 = C + Range * 1.1/12, R2 = C + Range * 1.1/6, R3 = C + Range * 1.1/4, R4 = C + Range * 1.1/2
    # Support levels: S1 = C - Range * 1.1/12, S2 = C - Range * 1.1/6, S3 = C - Range * 1.1/4, S4 = C - Range * 1.1/2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Key levels for trading: R3, R4, S3, S4 (stronger levels)
    camarilla_r3 = close_1w + range_1w * 1.1 / 4.0
    camarilla_r4 = close_1w + range_1w * 1.1 / 2.0
    camarilla_s3 = close_1w - range_1w * 1.1 / 4.0
    camarilla_s4 = close_1w - range_1w * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x average 12h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on Camarilla S3 retracement (mean reversion from strong level)
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Camarilla R3 retracement (mean reversion from strong level)
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume confirmation
            # Long on Camarilla R4 breakout, Short on Camarilla S4 breakout
            if volume_confirmed:
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals