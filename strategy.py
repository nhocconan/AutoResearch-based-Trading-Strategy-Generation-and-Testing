#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation
# Camarilla pivots from 12h provide structure aligned with 4h timeframe
# Volume confirmation (current 4h volume > 2.0x 20-period average) filters false breakouts
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# Works in bull/bear: price reacts to 12h structure, volume confirms validity
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "4h_12h_camarilla_volume_v1"
timeframe = "4h"
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
    if len(df_12h) < 25:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # Resistance levels: R1 = C + Range * 1.1/12, R2 = C + Range * 1.1/6, R3 = C + Range * 1.1/4, R4 = C + Range * 1.1/2
    # Support levels: S1 = C - Range * 1.1/12, S2 = C - Range * 1.1/6, S3 = C - Range * 1.1/4, S4 = C - Range * 1.1/2
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Key levels for trading: R3, R4, S3, S4 (stronger levels)
    camarilla_r3 = close_12h + range_12h * 1.1 / 4.0
    camarilla_r4 = close_12h + range_12h * 1.1 / 2.0
    camarilla_s3 = close_12h - range_12h * 1.1 / 4.0
    camarilla_s4 = close_12h - range_12h * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Pre-compute volume confirmation (20-period average for 4h)
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
        
        # Volume confirmation: current 4h volume > 2.0x average 4h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
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