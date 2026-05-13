#!/usr/bin/env python3
"""
12h_Camarilla_R1_S4_Breakout_1wTrend_Volume
Hypothesis: Use weekly Camarilla pivot levels (R1/S4) on 12h timeframe. Buy when price breaks above R1 with volume confirmation and weekly uptrend (price > weekly EMA50). Sell when price breaks below S4 with volume confirmation and weekly downtrend (price < weekly EMA50). Camarilla levels are effective in ranging markets and breakouts work well in trends. Weekly trend filter avoids counter-trend trades. Designed for 12h to limit trades (12-37/year) and avoid fee drag.
"""

name = "12h_Camarilla_R1_S4_Breakout_1wTrend_Volume"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1 / 12), S4 = close - (range * 1.1 / 2)
    r1 = close_1d + (range_1d * 1.1 / 12)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (no extra delay - levels known at daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (24-period = 12 days) for volume spike filter
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 24-period average
        vol_spike = volume[i] > 1.5 * vol_ma_24[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and weekly uptrend
            if close[i] > r1_aligned[i] and vol_spike and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 with volume spike and weekly downtrend
            elif close[i] < s4_aligned[i] and vol_spike and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4 or weekly trend turns down
            if close[i] < s4_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or weekly trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals