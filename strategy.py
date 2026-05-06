#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extreme readings with volume confirmation
# Long when Williams %R(14) crosses above -20 from below (oversold bounce) AND volume > 1.5 * avg_volume(20)
# Short when Williams %R(14) crosses below -80 from above (overbought rejection) AND volume > 1.5 * avg_volume(20)
# Exit when Williams %R returns to -50 (mean reversion center) or opposite extreme touched
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams %R identifies exhaustion points; volume confirmation ensures follow-through
# Works in bull markets (buying dips) and bear markets (selling rallies)

name = "12h_1dWilliamsR_Extreme_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed 1d bars for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    williams_r = -100 * (highest_high - close_1d) / denominator
    
    # Align 1d Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below (oversold bounce) with volume confirmation
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above (overbought rejection) with volume confirmation
            elif (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or touches -80 (reversal)
            if williams_r_aligned[i] >= -50 or williams_r_aligned[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or touches -20 (reversal)
            if williams_r_aligned[i] <= -50 or williams_r_aligned[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals