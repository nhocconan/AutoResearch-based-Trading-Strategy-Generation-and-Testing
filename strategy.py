#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
Hypothesis: Breakouts above R1 or below S1 on 4h chart with volume confirmation and 12h trend alignment capture institutional order flow while filtering false breakouts. Uses 12h trend for multi-timeframe confirmation to work in both bull and bear regimes.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
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
    
    # Get 12h data for trend filter and Camarilla levels from 12h (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12-period EMA on 12h close for trend filter
    ema12_12h = pd.Series(df_12h['close']).ewm(span=12, adjust=False, min_periods=12).mean().values
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, ema12_12h)
    
    # Calculate Camarilla levels from 12h data
    hl_range_12h = df_12h['high'] - df_12h['low']
    r1_12h = df_12h['close'] + hl_range_12h * 1.1 / 6
    s1_12h = df_12h['close'] - hl_range_12h * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h.values)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h.values)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above R1 with volume spike and above 12h EMA12 (uptrend)
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > trend_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and below 12h EMA12 (downtrend)
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < trend_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price drops below S1 or trend turns down
            if (close[i] < s1_aligned[i] or 
                close[i] < trend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above R1 or trend turns up
            if (close[i] > r1_aligned[i] or 
                close[i] > trend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals