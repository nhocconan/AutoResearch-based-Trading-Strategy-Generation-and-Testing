#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Williams %R extremes with volume confirmation
# Williams %R(14) from 1d identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought)
# Entry: Buy when %R crosses above -80 from below with volume > 1.5x 20-period average
# Entry: Sell when %R crosses below -20 from above with volume > 1.5x 20-period average
# Exit: Reverse signal or %R returns to neutral zone (-80 to -20)
# Works in bull/bear: mean reversion from extremes tends to hold across regimes
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_williamsr_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on daily timeframe
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (completed 1d bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on Williams %R returning to neutral or overbought
            if williams_r_aligned[i] > -50:  # Exit when back to neutral or overbought
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Williams %R returning to neutral or oversold
            if williams_r_aligned[i] < -50:  # Exit when back to neutral or oversold
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion entries from extremes with volume confirmation
            if volume_confirmed:
                # Long when Williams %R crosses above -80 from below (oversold recovery)
                if (williams_r_aligned[i] > -80 and 
                    i > 50 and 
                    williams_r_aligned[i-1] <= -80):
                    position = 1
                    signals[i] = 0.25
                # Short when Williams %R crosses below -20 from above (overbought rejection)
                elif (williams_r_aligned[i] < -20 and 
                      i > 50 and 
                      williams_r_aligned[i-1] >= -20):
                    position = -1
                    signals[i] = -0.25
    
    return signals