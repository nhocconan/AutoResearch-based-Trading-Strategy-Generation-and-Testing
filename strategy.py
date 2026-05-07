#!/usr/bin/env python3
"""
6h_Alligator_ElderRay_TripleFilter
Hypothesis: Combine Williams Alligator (trend filter) with Elder Ray (bull/bear power) and volume confirmation on 6h timeframe. Uses 1d trend to avoid counter-trend trades. Works in bull/bear markets by requiring alignment between 6h momentum and 1d trend. Targets 15-30 trades/year with low frequency to minimize fee impact.
"""

name = "6h_Alligator_ElderRay_TripleFilter"
timeframe = "6h"
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
    
    # Get 1d data for trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Williams Alligator on 6h (13,8,5 SMAs with future shifts)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Elder Ray Power on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d_for_power = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d_for_power
    bear_power = low_1d - ema_13_1d_for_power
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
        bullish_alignment = lips[i] > teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] < jaw[i]
        
        # 1d trend filter
        trend_up = close[i] > ema_13_1d_aligned[i]  # Using 6h close vs 1d EMA13
        trend_down = close[i] < ema_13_1d_aligned[i]
        
        # Elder Ray confirmation
        strong_bull_power = bull_power_aligned[i] > 0
        strong_bear_power = bear_power_aligned[i] < 0
        
        if position == 0:
            # Long: Alligator bullish + 1d uptrend + strong bull power + volume
            if (bullish_alignment and trend_up and strong_bull_power and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + 1d downtrend + strong bear power + volume
            elif (bearish_alignment and trend_down and strong_bear_power and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR bear power becomes strong
            if not bullish_alignment or strong_bear_power:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR bull power becomes strong
            if not bearish_alignment or strong_bull_power:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals