#!/usr/bin/env python3
# 6h_1D_WilliamsAlligator_ElderRay_Combined
# Hypothesis: Combines Williams Alligator (trend direction) with Elder Ray (bull/bear power) on daily timeframe.
# Alligator filters trend: price above/below teeth and lips aligned.
# Elder Ray measures strength: bull power (high-EMA13) and bear power (EMA13-low).
# Long when bullish trend + bull power > 0; short when bearish trend + bear power < 0.
# Uses volume spike confirmation to avoid false signals. Targets 12-37 trades/year on 6h timeframe.
# Works in bull markets via trend following and in bear markets via counter-trend power shifts.

name = "6h_1D_WilliamsAlligator_ElderRay_Combined"
timeframe = "6h"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator components on daily
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, shifted 3 bars)
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray components on daily
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d  # High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Low - EMA13 (negative when strong bear)
    
    # Align all indicators to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Williams Alligator trend conditions
        # Bullish trend: Lips > Teeth > Jaw (all aligned upward)
        bullish_trend = (lips_1d_aligned[i] > teeth_1d_aligned[i] and 
                         teeth_1d_aligned[i] > jaw_1d_aligned[i])
        # Bearish trend: Jaw > Teeth > Lips (all aligned downward)
        bearish_trend = (jaw_1d_aligned[i] > teeth_1d_aligned[i] and 
                         teeth_1d_aligned[i] > lips_1d_aligned[i])
        
        if position == 0:
            # LONG: Bullish trend + positive bull power + volume spike
            if (bullish_trend and 
                bull_power_1d_aligned[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish trend + negative bear power + volume spike
            elif (bearish_trend and 
                  bear_power_1d_aligned[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend turns bearish OR bull power turns negative
            if not bullish_trend or bull_power_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend turns bullish OR bear power turns positive
            if not bearish_trend or bear_power_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals