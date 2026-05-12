#!/usr/bin/env python3
# 12h_1w_1d_Camarilla_R4S4_Breakout_Trend_Filter
# Hypothesis: Uses weekly (1w) trend filter and daily (1d) Camarilla R4/S4 levels for breakout entries on 12h timeframe.
# Enters long when price breaks above daily R4 with 1w uptrend and volume confirmation.
# Enters short when price breaks below daily S4 with 1w downtrend and volume confirmation.
# Uses 1w EMA20 as trend filter to avoid counter-trend trades.
# Designed for low trade frequency (12-37 trades/year) to minimize fee drift.
# Works in bull/bear markets by following weekly trend while using daily Camarilla breakouts for precise entries.

name = "12h_1w_1d_Camarilla_R4S4_Breakout_Trend_Filter"
timeframe = "12h"
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
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = C + ((H-L) * 1.1/2)
    # S4 = C - ((H-L) * 1.1/2)
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align all indicators to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4 + 1w EMA20 uptrend + volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 + 1w EMA20 downtrend + volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4 OR closes below 1w EMA20
            if (close[i] < camarilla_s4_aligned[i]) or \
               (close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R4 OR closes above 1w EMA20
            if (close[i] > camarilla_r4_aligned[i]) or \
               (close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals