#!/usr/bin/env python3
name = "1d_WilliamsAlligator_ElderRay_WeeklyTrend_Filter"
timeframe = "1d"
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
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams Alligator on weekly (3 SMAs: 13, 8, 5)
    # Jaw (13-period, 8-bar shift)
    sma13_1w = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    jaw_1w = np.roll(sma13_1w, 8)  # shift forward 8 bars
    jaw_1w[:8] = np.nan  # first 8 invalid
    
    # Teeth (8-period, 5-bar shift)
    sma8_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    teeth_1w = np.roll(sma8_1w, 5)  # shift forward 5 bars
    teeth_1w[:5] = np.nan  # first 5 invalid
    
    # Lips (5-period, 3-bar shift)
    sma5_1w = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    lips_1w = np.roll(sma5_1w, 3)  # shift forward 3 bars
    lips_1w[:3] = np.nan  # first 3 invalid
    
    # Align Alligator components to daily
    jaw_1d = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1d = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1d = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Weekly trend: bullish when Lips > Teeth > Jaw
    weekly_bullish = (lips_1d > teeth_1d) & (teeth_1d > jaw_1d)
    weekly_bearish = (lips_1d < teeth_1d) & (teeth_1d < jaw_1d)
    
    # === DAILY DATA FOR ELDER RAY ===
    # Calculate 13-period EMA for Elder Ray
    ema13_1d = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = high - ema13_1d
    bear_power_1d = low - ema13_1d
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(lips_1d[i]) or
            np.isnan(bull_power_1d[i]) or np.isnan(bear_power_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish weekly trend + strong bull power + volume confirmation
            if weekly_bullish[i] and (bull_power_1d[i] > 0) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish weekly trend + strong bear power + volume confirmation
            elif weekly_bearish[i] and (bear_power_1d[i] < 0) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Weekly trend turns bearish OR bull power turns negative
            if weekly_bearish[i] or (bull_power_1d[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend turns bullish OR bear power turns positive
            if weekly_bullish[i] or (bear_power_1d[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals