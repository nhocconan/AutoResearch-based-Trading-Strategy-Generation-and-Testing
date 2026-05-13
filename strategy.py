#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla pivot levels from 1-day chart provide strong support/resistance.
Breakout above R1 with 1-day EMA34 uptrend and volume spike indicates bullish momentum.
Breakdown below S1 with 1-day EMA34 downtrend and volume spike indicates bearish momentum.
Designed for low trade frequency (~20-30/year) on 12h timeframe to avoid fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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
    
    # Get 1-day data once before loop (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # Where C, H, L are from previous day
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate Camarilla levels for each 1-day bar
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for 1-day close)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Trend filter: EMA34 on 1-day close
    ema_34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_12h
    downtrend_1d = close < ema_34_12h
    
    # Volume confirmation: > 1.5x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Close breaks above R1, 1-day uptrend, volume confirmation
            if close[i] > r1_12h[i] and uptrend_1d[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1, 1-day downtrend, volume confirmation
            elif close[i] < s1_12h[i] and downtrend_1d[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below S1 or trend fails
            if close[i] < s1_12h[i] or not uptrend_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above R1 or trend fails
            if close[i] > r1_12h[i] or not downtrend_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals