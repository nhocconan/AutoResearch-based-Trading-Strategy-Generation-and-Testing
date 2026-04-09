#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with volume confirmation
# Williams %R identifies overbought/oversold conditions on 1d timeframe
# Volume confirmation (current 6h volume > 1.8x 20-period average) filters false signals
# Works in bull/bear: mean reversion from extremes in ranging markets, 
#   continuation breakouts in trending markets with volume confirmation
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "6h_1d_williamsr_volume_v2"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Where period = 14 days
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x average 6h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on Williams %R returning above -20 (overbought exit)
            if williams_r_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Williams %R returning below -80 (oversold exit)
            if williams_r_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion entries with volume confirmation
            # Long when Williams %R < -80 (oversold) with volume confirmation
            # Short when Williams %R > -20 (overbought) with volume confirmation
            if volume_confirmed:
                if williams_r_aligned[i] < -80:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_aligned[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals