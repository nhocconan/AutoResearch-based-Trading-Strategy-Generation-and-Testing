#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1_S1_Breakout_Trend
Hypothesis: Camarilla R1/S1 breakouts on 4h, confirmed by 12h EMA trend and volume spike,
provide high-probability trend-following entries in both bull and bear markets.
Breakouts from key intraday support/resistance levels (R1/S1) capture momentum.
Volume spike confirms institutional interest. EMA trend filter ensures alignment with higher timeframe.
Target: 20-40 trades/year per symbol.
"""

name = "4h_12h_Camarilla_R1_S1_Breakout_Trend"
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
    
    # Calculate Camarilla levels from previous day
    # Use daily high/low/close from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    phigh = df_1d['high'].shift(1).values  # previous day high
    plow = df_1d['low'].shift(1).values    # previous day low
    pclose = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla R1 and S1 levels
    # R1 = c + (h-l)*1.1/12
    # S1 = c - (h-l)*1.1/12
    r1 = pclose + (phigh - plow) * 1.1 / 12
    s1 = pclose - (phigh - plow) * 1.1 / 12
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h trend filter: EMA 50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if Camarilla levels not available (first day)
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned values
        uptrend = uptrend_12h_aligned[i]
        downtrend = downtrend_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # LONG: price breaks above R1 + 12h uptrend + volume spike
            if close[i] > r1_aligned[i] and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + 12h downtrend + volume spike
            elif close[i] < s1_aligned[i] and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or trend reverses
            if close[i] < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or trend reverses
            if close[i] > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals