#!/usr/bin/env python3
"""
12h_200MA_Cross_With_1w_Trend_Filter
Hypothesis: 200-period EMA cross on 12h timeframe provides strong trend signals with low frequency. 
Weekly EMA 50 acts as trend filter to avoid counter-trend trades in choppy markets. 
Designed for very low trade frequency (<20/year) with high win rate in both bull and bear markets.
"""

name = "12h_200MA_Cross_With_1w_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 200 EMA for trend
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1-week trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 50-period average (to avoid low-volume false breaks)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if position == 0:
            # LONG: Price crosses above 200 EMA with weekly uptrend and volume confirmation
            if close[i] > ema_200[i] and close[i-1] <= ema_200[i-1]:
                if ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:  # Weekly EMA rising
                    if volume_confirm[i]:
                        signals[i] = 0.30
                        position = 1
            # SHORT: Price crosses below 200 EMA with weekly downtrend and volume confirmation
            elif close[i] < ema_200[i] and close[i-1] >= ema_200[i-1]:
                if ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:  # Weekly EMA falling
                    if volume_confirm[i]:
                        signals[i] = -0.30
                        position = -1
        elif position == 1:
            # EXIT LONG: Price crosses back below 200 EMA or weekly trend turns down
            if close[i] < ema_200[i] or ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses back above 200 EMA or weekly trend turns up
            if close[i] > ema_200[i] or ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals