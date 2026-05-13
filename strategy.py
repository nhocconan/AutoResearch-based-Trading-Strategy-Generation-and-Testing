#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_1wTrend
Hypothesis: In 12h timeframe, price breaking above Camarilla R1 or below S1 with volume confirmation indicates institutional breakout. Weekly trend filter ensures alignment with higher timeframe momentum, reducing false reversals. Works in bull markets by capturing continuation breaks and in bear markets by capturing sharp reversals with volume spikes. Uses discrete position sizing to minimize fee churn.
"""

name = "12h_1d_Camarilla_R1S1_Breakout_1wTrend"
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
    
    # Calculate 1d Camarilla levels (based on prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_r1[i] = close_1d[i-1] + range_ * 1.1 / 12
        camarilla_s1[i] = close_1d[i-1] - range_ * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day for availability)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = np.zeros_like(close_1w)
    for i in range(33, len(close_1w)):
        if i == 33:
            ema_34_1w[i] = np.mean(close_1w[:34])
        else:
            ema_34_1w[i] = (close_1w[i] * 2 / (34 + 1)) + (ema_34_1w[i-1] * (32 / (34 + 1)))
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: 20-period volume average
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 + volume spike + price above weekly EMA34
            if (close[i] > camarilla_r1_aligned[i] and vol_spike and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 + volume spike + price below weekly EMA34
            elif (close[i] < camarilla_s1_aligned[i] and vol_spike and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 or loses weekly uptrend
            if (close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 or loses weekly downtrend
            if (close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals