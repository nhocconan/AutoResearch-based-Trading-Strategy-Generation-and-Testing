#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R extreme readings with volume confirmation
# Williams %R identifies overbought/oversold conditions on weekly timeframe
# Extreme readings (%R < -80 for oversold, %R > -20 for overbought) signal potential reversals
# Volume confirmation (current 1d volume > 1.5x 20-period average) filters false signals
# Position size fixed at 0.25 to balance return and fee drag
# Target: 20-60 trades/year on 1d timeframe (80-240 total over 4 years)

name = "1d_1w_williamsr_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (14-period)
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w)
    # Handle division by zero (when high == low)
    williams_r_1w[highest_high_1w == lowest_low_1w] = -50
    
    # Align Williams %R to 1d timeframe
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC) - though less critical on 1d
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(williams_r_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Williams %R rises above -50 (momentum fading) or reaches overbought
            if williams_r_1w_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum fading) or reaches oversold
            if williams_r_1w_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Enter long on extreme oversold with volume confirmation
            if williams_r_1w_aligned[i] < -80:
                position = 1
                signals[i] = position_size
            # Enter short on extreme overbought with volume confirmation
            elif williams_r_1w_aligned[i] > -20:
                position = -1
                signals[i] = -position_size
    
    return signals