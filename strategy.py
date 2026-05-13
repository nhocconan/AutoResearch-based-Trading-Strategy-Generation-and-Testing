#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Breakouts above R1 or below S1 on 12h chart with volume confirmation and 1w trend alignment capture institutional order flow while filtering false breakouts. Uses 1w trend for multi-timeframe confirmation to work in both bull and bear regimes. Target 12-37 trades/year for low fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 20-period EMA on 1w close for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get 12h data for Camarilla levels (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels from 12h data
    hl_range_12h = df_12h['high'] - df_12h['low']
    r1_12h = df_12h['close'] + hl_range_12h * 1.1 / 6
    s1_12h = df_12h['close'] - hl_range_12h * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h.values)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h.values)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if position == 0:
            # LONG: break above R1 with volume spike and above 1w EMA20 (uptrend)
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and below 1w EMA20 (downtrend)
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price drops below S1 or trend turns down
            if (close[i] < s1_aligned[i] or 
                close[i] < trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above R1 or trend turns up
            if (close[i] > r1_aligned[i] or 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals