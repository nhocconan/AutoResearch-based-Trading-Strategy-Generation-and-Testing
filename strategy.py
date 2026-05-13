# 165083
#!/usr/bin/env python3
"""
4h_Donchian_Breakout_VolumeTrend_12hEMA
Hypothesis: Donchian channel breakouts (20-period) on 4h timeframe capture strong trends. 
Volume confirmation (>1.5x 20-period average) filters false breakouts. 
12h EMA(50) trend filter ensures alignment with higher timeframe momentum. 
Designed for low trade frequency (~20-50/year) to minimize fee drag. Works in both bull and bear markets by following the trend.
"""

name = "4h_Donchian_Breakout_VolumeTrend_12hEMA"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 12h EMA(50) trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above upper Donchian, volume confirmation, price above 12h EMA50 (uptrend)
            if (close[i] > high_20[i] and 
                volume_filter[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian, volume confirmation, price below 12h EMA50 (downtrend)
            elif (close[i] < low_20[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below lower Donchian (mean reversion) OR trend changes
            if (close[i] < low_20[i] or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above upper Donchian OR trend changes
            if (close[i] > high_20[i] or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals