#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Williams %R extremes + 1d volume spike confirmation
# Williams %R > -20 = overbought (short signal), < -80 = oversold (long signal)
# Only trade when 1d volume is above 2.0x its 20-period average to avoid low-volume false signals
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: mean reversion from extremes tends to work across regimes
# Williams %R is calculated on weekly data to reduce noise and avoid overtrading

name = "6h_1w_williamsr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period Williams %R on weekly data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1w) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume above 2.0x its 20-period average
        volume_confirmed = volume > 2.0 * avg_volume_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit long when Williams %R rises above -50 (exiting oversold territory)
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when Williams %R falls below -50 (exiting overbought territory)
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when Williams %R < -80 (oversold) with volume confirmation
            if williams_r_aligned[i] < -80 and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short when Williams %R > -20 (overbought) with volume confirmation
            elif williams_r_aligned[i] > -20 and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals