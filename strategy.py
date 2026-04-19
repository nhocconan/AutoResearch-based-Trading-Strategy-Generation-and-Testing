#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal from daily extremes with volume confirmation.
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# (%R > -10 for overbought, %R < -90 for oversold) with volume spike indicate high-probability
# mean reversion. Works in both bull (fade rallies) and bear (fade crashes) markets.
# Uses tight conditions to limit trades (~15-25/year) and avoid overtrading.
name = "6h_WilliamsR_Extreme_Reversal"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (high_14 - close_1d) / (high_14 - low_14)
    
    # Align to 6h timeframe (waits for prior day close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: reversal from oversold (%R < -90) with volume spike
            if (williams_r_aligned[i] < -90 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: reversal from overbought (%R > -10) with volume spike
            elif (williams_r_aligned[i] > -10 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit when %R returns to neutral (> -50) or reversal signal
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit when %R returns to neutral (< -50) or reversal signal
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals