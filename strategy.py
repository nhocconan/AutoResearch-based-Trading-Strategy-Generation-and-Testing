#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout with weekly trend filter and volume spike works in both bull and bear markets. 
Uses 4h timeframe for lower trade frequency (target: 20-50 trades/year). 
Weekly trend filter (EMA50) avoids counter-trend trades. Volume spike confirms breakout strength.
R1/S1 levels provide institutional support/resistance that works across market regimes.
"""

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
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
    
    # Volume spike: >2.0x 30-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for Camarilla calculation (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1w data for trend filter (weekly EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above weekly EMA50
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below weekly EMA50
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S1 and R1 OR closes below weekly EMA50
            if (close[i] > s1_aligned[i] and close[i] < r1_aligned[i]) or \
               close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between S1 and R1 OR closes above weekly EMA50
            if (close[i] > s1_aligned[i] and close[i] < r1_aligned[i]) or \
               close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals