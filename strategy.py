#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter
Hypothesis: Camarilla pivot levels from daily timeframe act as institutional support/resistance.
Breakouts above R1 (resistance 1) or below S1 (support 1) with volume confirmation and aligned
1d trend capture institutional flow while avoiding false breakouts in chop. Works in bull/bear via
trend filter and limits trades via strict entry conditions to reduce fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # R2 = C + ((H-L)*1.1/6)
    # R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12)
    # S2 = C - ((H-L)*1.1/6)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    # where C, H, L are previous day's close, high, low
    C = df_1d['close'].shift(1).values  # Previous day close
    H = df_1d['high'].shift(1).values   # Previous day high
    L = df_1d['low'].shift(1).values    # Previous day low
    
    # Avoid look-ahead: use previous day's data only
    R1 = C + ((H - L) * 1.1 / 12)
    S1 = C - ((H - L) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (available after previous day close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        if position == 0:
            # LONG: break above R1 with volume spike and above 1d EMA50 (uptrend)
            if (close[i] > R1_aligned[i] and 
                volume_spike[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and below 1d EMA50 (downtrend)
            elif (close[i] < S1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below S1 or trend turns down
            if (close[i] < S1_aligned[i] or 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above R1 or trend turns up
            if (close[i] > R1_aligned[i] or 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals