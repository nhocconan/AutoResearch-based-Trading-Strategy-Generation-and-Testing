#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Donchian(20) breakouts filtered by 12h EMA(20) trend and volume spikes capture
sustained moves in both bull and bear markets, while avoiding false signals in ranging conditions.
Targets 20-50 trades/year with low turnover to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 10-period average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 12h EMA20
            uptrend = close[i] > ema_20_12h_aligned[i]
            downtrend = close[i] < ema_20_12h_aligned[i]
            
            # Long: price breaks above upper band + uptrend + volume spike
            if (close[i] > high_20[i] and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower band + downtrend + volume spike
            elif (close[i] < low_20[i] and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals