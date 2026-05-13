#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeS
Hypothesis: 1h breakouts at Camarilla R1/S1 with 4h trend confirmation and volume spikes capture momentum with controlled frequency. Uses 4h for direction (trend) and 1h for precise entry timing, targeting 60-150 trades over 4 years. Works in bull/bear by following 4h trend only.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeS"
timeframe = "1h"
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
    
    # Get 4h data for trend filter and Camarilla levels (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels for each 4h bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    hl_range = df_4h['high'] - df_4h['low']
    r1 = df_4h['close'] + hl_range * 1.1 / 12
    s1 = df_4h['close'] - hl_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1.values)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above R1 with volume spike and above 4h EMA50 (uptrend)
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > trend_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: break below S1 with volume spike and below 4h EMA50 (downtrend)
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < trend_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price drops below S1 or trend turns down
            if (close[i] < s1_aligned[i] or 
                close[i] < trend_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price rises above R1 or trend turns up
            if (close[i] > r1_aligned[i] or 
                close[i] > trend_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals