#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation
# Camarilla pivots identify key intraday support/resistance levels (R3,S3,R4,S4)
# In ranging markets: fade extreme levels (R3/S3) with mean reversion
# In trending markets: breakout continuation at R4/S4 levels
# Volume confirmation filters false breakouts/breakdowns
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion in ranges, breakout continuation in trends

name = "6h_12h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3_12h = pivot_12h + range_12h * 1.1 / 4.0
    s3_12h = pivot_12h - range_12h * 1.1 / 4.0
    r4_12h = pivot_12h + range_12h * 1.1 / 2.0
    s4_12h = pivot_12h - range_12h * 1.1 / 2.0
    
    # Calculate 12h volume moving average for confirmation
    volume_s_12h = pd.Series(volume_12h)
    vol_ma_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 12h average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if price breaks below S3 (mean reversion failure) or reaches R4 (take profit)
            if close[i] < s3_12h_aligned[i] or close[i] > r4_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above R3 (mean reversion failure) or reaches S4 (take profit)
            if close[i] > r3_12h_aligned[i] or close[i] < s4_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long at S3 with volume confirmation (mean reversion long)
            if close[i] <= s3_12h_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short at R3 with volume confirmation (mean reversion short)
            elif close[i] >= r3_12h_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
            # Enter long on breakout above R4 with volume confirmation
            elif close[i] > r4_12h_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short on breakdown below S4 with volume confirmation
            elif close[i] < s4_12h_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals