#!/usr/bin/env python3
"""
6h Williams %R Mean Reversion + 1D Trend Filter + Volume Spike v1
Hypothesis: In both bull and bear markets, price tends to revert to the mean after extreme moves. Williams %R identifies overbought/oversold conditions, while 1D EMA trend ensures we trade in the direction of higher-timeframe momentum. Volume spikes confirm the exhaustion of the move. This combination works in ranging markets (reversion) and trending markets (pullbacks in trend direction), targeting 15-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_r_mean_reversion_1d_trend_volume_v1"
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
    
    # 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1D EMA(50) for trend filter
    ema_50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    williams_r = -100 * (highest_high - close) / hl_range
    
    # 6h Volume spike (>2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns above -20 (overbought) or trend reverses
            if williams_r[i] >= -20 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -80 (oversold) or trend reverses
            if williams_r[i] <= -80 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: oversold (-80 to -100) + volume spike + uptrend (price > EMA)
            if (williams_r[i] <= -80 and 
                vol_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: overbought (0 to -20) + volume spike + downtrend (price < EMA)
            elif (williams_r[i] >= -20 and 
                  vol_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals