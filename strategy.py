#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R breakout with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; breakout from extreme levels
# with volume confirmation captures strong momentum moves. 1d EMA(34) ensures alignment
# with higher timeframe trend to avoid counter-trend trades. Works in bull/bear markets
# by following 1d trend direction for entries. Target: 50-150 total trades over 4 years
# (12-37/year) to balance opportunity and fee drag.

name = "6h_WilliamsR14_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Williams %R and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Where Highest High and Lowest Low are over the past 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams %R breakout signals with 1d trend filter
        # Long: Williams %R crosses above -80 (oversold) + price above 1d EMA34 + volume spike
        # Short: Williams %R crosses below -20 (overbought) + price below 1d EMA34 + volume spike
        if position == 0:
            # Check for crossover: previous value below threshold, current value above/below
            prev_williams = williams_r_aligned[i-1] if i > 0 else -50
            
            if (prev_williams <= -80 and williams_r_aligned[i] > -80 and 
                close[i] > ema_34_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (prev_williams >= -20 and williams_r_aligned[i] < -20 and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR price below 1d EMA34
            prev_williams = williams_r_aligned[i-1] if i > 0 else -50
            if (prev_williams < -20 and williams_r_aligned[i] >= -20) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR price above 1d EMA34
            prev_williams = williams_r_aligned[i-1] if i > 0 else -50
            if (prev_williams > -80 and williams_r_aligned[i] <= -80) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals