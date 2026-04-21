#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeFilter_v1
Hypothesis: Breakout of Camarilla R1/S1 levels on 4h with 1d trend alignment (EMA34) and volume confirmation.
Uses 4h timeframe to reduce trade frequency, 1d EMA for trend filter, and volume spike for confirmation.
Designed to work in both bull/bear markets by aligning with higher timeframe trend.
Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for trend (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 4h data once for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar's OHLC for Camarilla calculation
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R2, S2
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    r2 = prev_close + rang * 2.0 / 12
    s2 = prev_close - rang * 2.0 / 12
    
    # Align Camarilla levels to 4h timeframe (already aligned since we're using 4h data)
    # No need to align as we're calculating on the same timeframe
    r1_4h = r1
    s1_4h = s1
    r2_4h = r2
    s2_4h = s2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(r2_4h[i]) or np.isnan(s2_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above R1 with 1d uptrend and volume
            if (price > r1_4h[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and  # 1d EMA rising
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 with 1d downtrend and volume
            elif (price < s1_4h[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and  # 1d EMA falling
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below S1 or reach R2
            if price < s1_4h[i] or price > r2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R1 or reach S2
            if price > r1_4h[i] or price < s2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0