#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Williams %R + Volume Spike Reversal
# Hypothesis: In 6h timeframe, extreme Williams %R readings (< -80 or > -20) 
# combined with volume spikes (>2x 20-period average) signal exhaustion and 
# impending reversals. Uses daily Williams %R for higher timeframe context to 
# avoid counter-trend traps. Works in both bull/bear markets by fading extremes 
# rather than chasing trends. Target: 20-35 trades/year (80-140 total over 4 years).

name = "6h_daily_williamsr_volume_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R on daily data (14-period)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - daily_close) / (highest_high - lowest_low)) * -100, 
                          -50)  # Neutral when range is zero
    
    # Williams %R signals: > -20 = overbought, < -80 = oversold
    williams_r_overbought = williams_r > -20
    williams_r_oversold = williams_r < -80
    
    # Align daily Williams %R signals to 6h timeframe
    williams_r_overbought_aligned = align_htf_to_ltf(prices, df_daily, williams_r_overbought.astype(float))
    williams_r_oversold_aligned = align_htf_to_ltf(prices, df_daily, williams_r_oversold.astype(float))
    
    # Volume filter on 6h: volume > 2x 20-period average (strong spike)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(williams_r_overbought_aligned[i]) or 
            np.isnan(williams_r_oversold_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns from oversold or volume spike fades
            if williams_r_oversold_aligned[i] == False or volume_spike[i] == False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: Williams %R returns from overbought or volume spike fades
            if williams_r_overbought_aligned[i] == False or volume_spike[i] == False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Fade extremes with volume confirmation
            if williams_r_overbought_aligned[i] and volume_spike[i]:
                # Short at daily overbought + volume spike (expecting reversal down)
                position = -1
                signals[i] = -0.25
            elif williams_r_oversold_aligned[i] and volume_spike[i]:
                # Long at daily oversold + volume spike (expecting reversal up)
                position = 1
                signals[i] = 0.25
    
    return signals