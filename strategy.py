#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation
# Fade at R3/S3 levels (mean reversion), breakout continuation at R4/S4 levels
# Uses 1d Camarilla levels calculated from previous day's range
# Works in both trending and ranging markets: mean reversion in ranges, breakout in trends
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_camarilla_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 2:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i > 0:  # Need previous day's data
            h = high_1d[i-1]
            l = low_1d[i-1]
            c = close_1d[i-1]
            diff = h - l
            camarilla_r4[i] = c + diff * 1.1 / 2
            camarilla_r3[i] = c + diff * 1.1 / 4
            camarilla_s3[i] = c - diff * 1.1 / 4
            camarilla_s4[i] = c - diff * 1.1 / 2
        else:
            # First day: use same day's data (will be overridden by alignment)
            camarilla_r4[i] = close_1d[i]
            camarilla_r3[i] = close_1d[i]
            camarilla_s3[i] = close_1d[i]
            camarilla_s4[i] = close_1d[i]
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            # Exit if price reaches R4 (take profit) or breaks below S3 (stop)
            if close[i] >= r4_aligned[i] or close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit if price reaches S4 (take profit) or breaks above R3 (stop)
            if close[i] <= s4_aligned[i] or close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long setup: price near S3 with bounce (mean reversion)
            if (close[i] <= s3_aligned[i] * 1.005 and  # Within 0.5% of S3
                close[i] > close[i-1] and  # Price bouncing up
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price near R3 with rejection (mean reversion)
            elif (close[i] >= r3_aligned[i] * 0.995 and  # Within 0.5% of R3
                  close[i] < close[i-1] and  # Price rejecting down
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            # Breakout continuation: strong volume break above R4 or below S4
            elif (close[i] > r4_aligned[i] and 
                  close[i-1] <= r4_aligned[i-1] and  # Just broke above
                  volume[i] > volume_threshold[i] * 1.5):  # Strong volume
                signals[i] = 0.25
                position = 1
            elif (close[i] < s4_aligned[i] and 
                  close[i-1] >= s4_aligned[i-1] and  # Just broke below
                  volume[i] > volume_threshold[i] * 1.5):  # Strong volume
                signals[i] = -0.25
                position = -1
    
    return signals