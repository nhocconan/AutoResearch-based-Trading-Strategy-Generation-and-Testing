#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R1, S1) from the 12h chart act as strong support/resistance.
A breakout above R1 with 12h uptrend and volume spike signals a long entry.
A breakdown below S1 with 12h downtrend and volume spike signals a short entry.
This strategy targets 15-35 trades/year per symbol by requiring confluence of price level,
trend, and volume confirmation, reducing whipsaw in sideways markets.
Works in both bull and bear markets by following the 12h trend direction.
"""

name = "4h_12h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Get 12h data for Camarilla pivot calculation and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for each 12h bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    cam_r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12
    cam_s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar to close)
    cam_r1_aligned = align_htf_to_ltf(prices, df_12h, cam_r1_12h)
    cam_s1_aligned = align_htf_to_ltf(prices, df_12h, cam_s1_12h)
    
    # 12h trend: 34 EMA
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_12h = close_12h > ema_34_12h
    downtrend_12h = close_12h < ema_34_12h
    
    # Align 12h trend to 4h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume spike: volume > 2.0 * 20-period average (to reduce false signals)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get aligned values for current bar
        r1 = cam_r1_aligned[i]
        s1 = cam_s1_aligned[i]
        uptrend = uptrend_12h_aligned[i]
        downtrend = downtrend_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # LONG: price breaks above R1, 12h uptrend, volume spike
            if close[i] > r1 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1, 12h downtrend, volume spike
            elif close[i] < s1 and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or 12h trend turns down
            if close[i] < s1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or 12h trend turns up
            if close[i] > r1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals