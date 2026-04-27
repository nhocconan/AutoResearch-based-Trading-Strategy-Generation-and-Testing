#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA trend filter and volume spike.
# Williams %R measures overbought/oversold conditions.
# Williams %R < -80 = oversold (potential long)
# Williams %R > -20 = overbought (potential short)
# Strategy: Enter long when Williams %R crosses above -80 in uptrend with volume spike
# Enter short when Williams %R crosses below -20 in downtrend with volume spike
# Uses 1d EMA50 as trend filter to avoid counter-trend trades
# Designed for ~15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    williams_r = -100 * (highest_high - close) / hl_range
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R crosses above -80 (from below) in uptrend with volume
        if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
            close[i] > ema50_1d_aligned[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R crosses below -20 (from above) in downtrend with volume
        elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
              close[i] < ema50_1d_aligned[i] and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or loss of momentum
        elif position == 1 and (williams_r[i] < -50 or close[i] < ema50_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (williams_r[i] > -50 or close[i] > ema50_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold current position
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0