#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_1dTrend_Volume
Uses 1d Camarilla R1/S1 levels as key support/resistance on 12h timeframe.
Enters long when price breaks above R1 with 1d uptrend and volume confirmation.
Enters short when price breaks below S1 with 1d downtrend and volume confirmation.
Uses 1d EMA50 as trend filter to avoid counter-trend trades.
Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drift.
Works in bull/bear markets by following 1d trend while using 1d Camarilla breakouts for precise entries.
"""

name = "12h_1d_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Volume spike: >1.5x 30-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla pivot levels and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + 1d EMA50 uptrend + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S1 + 1d EMA50 downtrend + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR closes below 1d EMA50
            if (close[i] < camarilla_s1_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR closes above 1d EMA50
            if (close[i] > camarilla_r1_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals