#!/usr/bin/env python3
# 12h_Three_Month_High_Low_Breakout
# Hypothesis: Breakout above 3-month high (63-day high) or below 3-month low (63-day low)
# on 12h timeframe with volume confirmation and 1d trend filter. The 3-month high/low
# captures major structural levels that act as support/resistance. In bull markets,
# breaks above 3-month high signal continuation; in bear markets, breaks below
# 3-month low signal continuation. Volume filter reduces false breakouts. Uses
# 1d EMA34 for trend filter to align with major trend. Target: 25-40 trades/year.

name = "12h_Three_Month_High_Low_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for 3-month high/low and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 63:
        return np.zeros(n)
    
    # 3-month high (63-day high) and low (63-day low)
    high_63 = pd.Series(df_1d['high']).rolling(window=63, min_periods=63).max().values
    low_63 = pd.Series(df_1d['low']).rolling(window=63, min_periods=63).min().values
    
    # Align 3-month levels to 12h timeframe
    high_63_12h = align_htf_to_ltf(prices, df_1d, high_63)
    low_63_12h = align_htf_to_ltf(prices, df_1d, low_63)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(63, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(high_63_12h[i]) or np.isnan(low_63_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above 3-month high in daily uptrend with volume
            if close[i] > high_63_12h[i] and close[i] > ema_34_12h[i] and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: break below 3-month low in daily downtrend with volume
            elif close[i] < low_63_12h[i] and close[i] < ema_34_12h[i] and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns to 3-month low or trend reverses
            if close[i] < low_63_12h[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns to 3-month high or trend reverses
            if close[i] > high_63_12h[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals