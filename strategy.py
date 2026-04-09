#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w/1d Camarilla pivot levels with volume confirmation
# Weekly Camarilla pivots provide major structure, daily for intermediate levels
# Volume confirmation (current 12h volume > 1.8x 20-period average) filters false breakouts
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# Works in bull/bear: price reacts to weekly structure, volume confirms validity
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "12h_1w_1d_camarilla_volume_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Camarilla pivot levels (stronger structure)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    camarilla_w_r3 = close_1w + range_1w * 1.1 / 4.0  # Weekly R3
    camarilla_w_r4 = close_1w + range_1w * 1.1 / 2.0  # Weekly R4
    camarilla_w_s3 = close_1w - range_1w * 1.1 / 4.0  # Weekly S3
    camarilla_w_s4 = close_1w - range_1w * 1.1 / 2.0  # Weekly S4
    
    # Calculate daily Camarilla pivot levels (intermediate structure)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_d_r3 = close_1d + range_1d * 1.1 / 4.0  # Daily R3
    camarilla_d_r4 = close_1d + range_1d * 1.1 / 2.0  # Daily R4
    camarilla_d_s3 = close_1d - range_1d * 1.1 / 4.0  # Daily S3
    camarilla_d_s4 = close_1d - range_1d * 1.1 / 2.0  # Daily S4
    
    # Align weekly Camarilla levels to 12h timeframe
    w_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_w_r3)
    w_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_w_r4)
    w_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_w_s3)
    w_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_w_s4)
    
    # Align daily Camarilla levels to 12h timeframe
    d_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_r3)
    d_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_r4)
    d_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_s3)
    d_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_s4)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(w_r3_aligned[i]) or np.isnan(w_r4_aligned[i]) or
            np.isnan(w_s3_aligned[i]) or np.isnan(w_s4_aligned[i]) or
            np.isnan(d_r3_aligned[i]) or np.isnan(d_r4_aligned[i]) or
            np.isnan(d_s3_aligned[i]) or np.isnan(d_s4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x average 12h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on daily S3 retracement (mean reversion from daily strong level)
            if close[i] < d_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on daily R3 retracement (mean reversion from daily strong level)
            if close[i] > d_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume confirmation
            # Require BOTH weekly and daily levels to align for stronger signal
            # Long on break above weekly R4 AND daily R4
            # Short on break below weekly S4 AND daily S4
            if volume_confirmed:
                if close[i] > w_r4_aligned[i] and close[i] > d_r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < w_s4_aligned[i] and close[i] < d_s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals