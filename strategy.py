#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R extreme + volume confirmation + price above/below 1w EMA50
# Williams %R < -80 = oversold (long), > -20 = overbought (short) on weekly timeframe
# Enter only when price confirms with 1d close beyond EMA50 and volume > 1.5x 20-day average
# Works in bull/bear: mean reversion from extremes, volume confirms conviction
# Target: 15-25 trades/year, discrete sizing 0.25 to minimize fee drag

name = "1d_1w_williamsr_extreme_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1w) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range=0
    )
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume (20-period)
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or price below EMA50
            if williams_r_aligned[i] > -50 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or price above EMA50
            if williams_r_aligned[i] < -50 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Williams %R < -80 (oversold) + price above EMA50 + volume confirmation
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Enter short: Williams %R > -20 (overbought) + price below EMA50 + volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals