#!/usr/bin/env python3
"""
4h_ema_touch_1d_trend_volume_v1
Hypothesis: Price touching EMA(21) on 4h with daily trend filter and volume confirmation captures pullbacks in trending markets. Works in bull (buy EMA support) and bear (sell EMA resistance) by following daily trend. Low-frequency entries reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_touch_1d_trend_volume_v1"
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
    
    # 4h EMA(21) for entry
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Daily trend filter: EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema_50_d = df_1d['close'].ewm(span=50, adjust=False).mean().values
    ema_50_d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_d)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if required data not available
        if (np.isnan(ema_21[i]) or np.isnan(ema_50_d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA(21) or trend turns bearish
            if close[i] < ema_21[i] or close[i] < ema_50_d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above EMA(21) or trend turns bullish
            if close[i] > ema_21[i] or close[i] > ema_50_d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches EMA(21) from above with volume and bullish daily trend
            if (low[i] <= ema_21[i] * 1.001 and high[i] >= ema_21[i] * 0.999 and  # touching EMA
                vol_confirm and 
                close[i] > ema_50_d_aligned[i]):  # daily uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: price touches EMA(21) from below with volume and bearish daily trend
            elif (low[i] <= ema_21[i] * 1.001 and high[i] >= ema_21[i] * 0.999 and  # touching EMA
                  vol_confirm and 
                  close[i] < ema_50_d_aligned[i]):  # daily downtrend
                position = -1
                signals[i] = -0.25
    
    return signals