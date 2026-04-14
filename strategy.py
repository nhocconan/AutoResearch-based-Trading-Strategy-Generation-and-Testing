#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation.
# Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels (trend following).
# Uses volume spike (>1.5x 20-period average) to confirm both fade and breakout signals.
# Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
# Works in bull/bear: mean reversion works in range, breakout works in trending markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # R3 = C + (Range * 1.1/4)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all indicators to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r4_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 1d volume vs 20-period average
        volume_ratio = volume_1d_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entry signals with volume confirmation
            # Fade at R3/S3 (mean reversion) - sell at R3, buy at S3
            # Breakout at R4/S4 (trend following) - buy above R4, sell below S4
            
            # Long signals:
            # 1. Breakout: price > R4 with volume confirmation
            # 2. Fade: price < S3 with volume confirmation (mean reversion bounce)
            if ((close[i] > r4_1d_aligned[i] or close[i] < s3_1d_aligned[i]) and 
                volume_ratio > 1.5):
                if close[i] > r4_1d_aligned[i]:
                    # Breakout long
                    position = 1
                    signals[i] = position_size
                else:
                    # Fade long (bounce from S3)
                    position = 1
                    signals[i] = position_size
            # Short signals:
            # 1. Breakout: price < S4 with volume confirmation
            # 2. Fade: price > R3 with volume confirmation (mean reversion rejection)
            elif ((close[i] < s4_1d_aligned[i] or close[i] > r3_1d_aligned[i]) and 
                  volume_ratio > 1.5):
                if close[i] < s4_1d_aligned[i]:
                    # Breakout short
                    position = -1
                    signals[i] = -position_size
                else:
                    # Fade short (rejection at R3)
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches opposite level or volume drops
            if (close[i] < s3_1d_aligned[i] or  # Reached S3 (target for fade) or below
                volume_ratio < 1.0):  # Volume dried up
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches opposite level or volume drops
            if (close[i] > r3_1d_aligned[i] or  # Reached R3 (target for fade) or above
                volume_ratio < 1.0):  # Volume dried up
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Camarilla_Pivot_FadeBreakout_Volume_v1"
timeframe = "6h"
leverage = 1.0