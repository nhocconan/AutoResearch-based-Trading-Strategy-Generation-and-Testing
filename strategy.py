#!/usr/bin/env python3
"""
4h_Donchian_Breakout_12hEMA34_Volume
Hypothesis: Donchian(20) breakout with 12-hour EMA34 trend filter and volume confirmation captures
trend continuation moves in both bull and bear markets. Breakouts from 20-period price channels
signal momentum shifts, while EMA34 filters counter-trend noise. Volume >1.5x average confirms
institutional participation. Target: 20-40 trades/year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12-hour EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_4h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_12h_4h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_ok = volume_filter[i]
        trend = ema_12h_4h[i]
        
        if position == 0:
            # Long: break above upper band with volume in uptrend
            if price > upper and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume in downtrend
            elif price < lower and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price closes below lower band (reversal) or trend fails
            if price < lower or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price closes above upper band (reversal) or trend fails
            if price > upper or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0