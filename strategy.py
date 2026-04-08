#!/usr/bin/env python3
"""
6h Elder Ray Index with 12h Trend Filter and Volume Confirmation
Hypothesis: Elder Ray (bull/bear power) measures trend strength via EMA deviation. 
Combines with 12h EMA trend filter and volume to avoid whipsaws. 
Works in bull/bear by aligning with higher timeframe trend.
Targets 15-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_12h_trend_volume_v1"
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
    
    # 12h EMA(13) for Elder Ray and trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_13_12h = pd.Series(df_12h['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    
    # Elder Ray Components: Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema_13_12h_aligned
    bear_power = low - ema_13_12h_aligned
    
    # Volume filter (>1.3x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bear power turns positive OR trend turns bearish
            if (bear_power[i] > 0 or close[i] < ema_13_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull power turns negative OR trend turns bullish
            if (bull_power[i] < 0 or close[i] > ema_13_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: strong bull power, uptrend, volume
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_13_12h_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: strong bear power, downtrend, volume
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  close[i] < ema_13_12h_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals