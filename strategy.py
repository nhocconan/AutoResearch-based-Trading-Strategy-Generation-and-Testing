#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions.
# Williams %R < -80 = oversold (long opportunity)
# Williams %R > -20 = overbought (short opportunity)
# Strategy: Enter long when Williams %R crosses above -80 in uptrend (price > 1d EMA)
# Enter short when Williams %R crosses below -20 in downtrend (price < 1d EMA)
# Volume spike confirms institutional participation.
# Designed for ~12-37 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denom = highest_high - lowest_low
    denom = np.where(denom == 0, 1e-10, denom)
    
    williams_r = -100 * (highest_high - close) / denom
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R crossover signals
        williams_r_prev = williams_r[i-1] if i > 0 else williams_r[i]
        
        # Long signal: Williams %R crosses above -80 (oversold to neutral) in uptrend
        if (williams_r_prev <= -80 and williams_r[i] > -80 and 
            close[i] > ema34_1d_aligned[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        
        # Short signal: Williams %R crosses below -20 (overbought to neutral) in downtrend
        elif (williams_r_prev >= -20 and williams_r[i] < -20 and 
              close[i] < ema34_1d_aligned[i] and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        
        # Exit conditions: reverse signal or loss of trend/volume
        elif position == 1:
            # Exit long if Williams %R goes above -20 (overbought) or trend breaks
            if williams_r[i] >= -20 or close[i] <= ema34_1d_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        
        elif position == -1:
            # Exit short if Williams %R goes below -80 (oversold) or trend breaks
            if williams_r[i] <= -80 or close[i] >= ema34_1d_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
        
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0