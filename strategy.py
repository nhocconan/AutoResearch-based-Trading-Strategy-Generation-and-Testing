#!/usr/bin/env python3
# 12h_1d_1w_Camarilla_R4S4_Breakout_Trend_Filter
# Hypothesis: Uses 1d and 1w Camarilla R4/S4 levels as key support/resistance on 12h timeframe.
# Enters long when price breaks above weekly R4 or daily R4 with 12h uptrend and volume confirmation.
# Enters short when price breaks below weekly S4 or daily S4 with 12h downtrend and volume confirmation.
# Uses 12h EMA100 as trend filter to avoid counter-trend trades.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by following 12h trend while using multi-timeframe Camarilla breakouts for precise entries.

name = "12h_1d_1w_Camarilla_R4S4_Breakout_Trend_Filter"
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
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each day and week
    # R4 = C + ((H-L) * 1.1/2)
    # S4 = C - ((H-L) * 1.1/2)
    camarilla_r4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    camarilla_r4_1w = close_1w + ((high_1w - low_1w) * 1.1 / 2)
    camarilla_s4_1w = close_1w - ((high_1w - low_1w) * 1.1 / 2)
    
    # 12h EMA100 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    ema_100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(camarilla_r4_1d_aligned[i]) or
            np.isnan(camarilla_s4_1d_aligned[i]) or
            np.isnan(camarilla_r4_1w_aligned[i]) or
            np.isnan(camarilla_s4_1w_aligned[i]) or
            np.isnan(ema_100_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4 (daily or weekly) + 12h EMA100 uptrend + volume spike
            if ((close[i] > camarilla_r4_1d_aligned[i] or close[i] > camarilla_r4_1w_aligned[i]) and 
                close[i] > ema_100_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 (daily or weekly) + 12h EMA100 downtrend + volume spike
            elif ((close[i] < camarilla_s4_1d_aligned[i] or close[i] < camarilla_s4_1w_aligned[i]) and 
                  close[i] < ema_100_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4 (daily or weekly) OR closes below 12h EMA100
            if (close[i] < camarilla_s4_1d_aligned[i]) or \
               (close[i] < camarilla_s4_1w_aligned[i]) or \
               (close[i] < ema_100_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R4 (daily or weekly) OR closes above 12h EMA100
            if (close[i] > camarilla_r4_1d_aligned[i]) or \
               (close[i] > camarilla_r4_1w_aligned[i]) or \
               (close[i] > ema_100_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals