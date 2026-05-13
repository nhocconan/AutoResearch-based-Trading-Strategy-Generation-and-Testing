# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily chart combined with 1d EMA trend filter and volume confirmation
provide high-probability breakout entries in both bull and bear markets. Uses tight entry conditions to limit trades
to ~25-40 per year, reducing fee drag while capturing momentum moves.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily HTF data for Camarilla pivots and trend ---
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels: R1, S1 (using previous day's OHLC)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are previous day's close, high, low
    if len(df_1d) < 2:
        return np.zeros(n)
    phigh = df_1d['high'].shift(1).values  # previous day high
    plow = df_1d['low'].shift(1).values    # previous day low
    pclose = df_1d['close'].shift(1).values # previous day close
    r1 = pclose + (phigh - plow) * 1.1 / 12
    s1 = pclose - (phigh - plow) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend = close > ema_34_4h
    downtrend = close < ema_34_4h
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above R1, uptrend, volume confirmation
            if close[i] > r1_4h[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, downtrend, volume confirmation
            elif close[i] < s1_4h[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below S1 or trend reverses
            if close[i] < s1_4h[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above R1 or trend reverses
            if close[i] > r1_4h[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals