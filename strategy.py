#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike
# Long when: price > Alligator Jaw AND Bull Power > 0 AND Bear Power < 0 AND volume > 1.5x 20 EMA volume
# Short when: price < Alligator Jaw AND Bear Power > 0 AND Bull Power < 0 AND volume > 1.5x 20 EMA volume
# Uses weekly trend filter: only long when weekly close > weekly EMA34, only short when weekly close < weekly EMA34
# Discrete sizing 0.25 to limit fee drag. Target: 15-30 trades/year per symbol.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Williams Alligator identifies trend, Elder Ray measures bull/bear power, volume confirms conviction.
# Weekly filter prevents counter-trend trades and reduces whipsaw.

name = "1d_WilliamsAlligator_ElderRay_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator (13,8,5 SMAs shifted)
    # Jaw: 13-period SMA shifted 8 bars
    # Teeth: 8-period SMA shifted 5 bars  
    # Lips: 5-period SMA shifted 3 bars
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw (13,8)
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8,5)
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5,3)
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator Jaw is the reference line
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    
    # Calculate Elder Ray Power
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when weekly close > weekly EMA34, downtrend when weekly close < weekly EMA34
    uptrend_1w = close_1w > ema34_1w
    downtrend_1w = close_1w < ema34_1w
    
    # Align weekly trend to daily timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND Bull Power > 0 AND Bear Power < 0 AND volume spike AND weekly uptrend
            if (close[i] > jaw_1d_aligned[i] and 
                bull_power_1d_aligned[i] > 0 and 
                bear_power_1d_aligned[i] < 0 and 
                volume_spike[i] and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw AND Bear Power > 0 AND Bull Power < 0 AND volume spike AND weekly downtrend
            elif (close[i] < jaw_1d_aligned[i] and 
                  bear_power_1d_aligned[i] > 0 and 
                  bull_power_1d_aligned[i] < 0 and 
                  volume_spike[i] and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Jaw OR Bear Power > 0 (bulls losing control) OR weekly trend changes to downtrend
            if (close[i] < jaw_1d_aligned[i] or 
                bear_power_1d_aligned[i] > 0 or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Jaw OR Bull Power > 0 (bears losing control) OR weekly trend changes to uptrend
            if (close[i] > jaw_1d_aligned[i] or 
                bull_power_1d_aligned[i] > 0 or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals