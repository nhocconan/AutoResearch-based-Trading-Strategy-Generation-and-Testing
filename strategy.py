#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeFilter
Hypothesis: Breakouts from 20-period Donchian channels on 4h timeframe, when aligned with 
daily trend (close > EMA50) and accompanied by volume spikes (>2x 20-period average), 
capture strong momentum moves. This works in bull markets via upward breakouts and 
in bear markets via downward breakdowns. Position size 0.25 limits risk while 
volume and trend filters reduce false signals. Target: 20-40 trades/year.
"""

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Breakout above upper Donchian with volume and uptrend
            if (close[i] > high_max[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below lower Donchian with volume and downtrend
            elif (close[i] < low_min[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below upper Donchian or trend reverses
            if (close[i] < high_max[i]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above lower Donchian or trend reverses
            if (close[i] > low_min[i]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals