#!/usr/bin/env python3
"""
4h_Williams_Alligator_DMI_Trend_Filter
Hypothesis: Williams Alligator defines the trend (teeth above/below lips), DMI (ADX>25) confirms strength.
Long when green alignment (jaws<teeth<lips) + ADX>25, short when red alignment (jaws>teeth>lips) + ADX>25.
Uses 13/8/5 SMAs with proper alignment. Works in both bull (strong uptrend) and bear (strong downtrend).
Target: 20-30 trades per year (~80-120 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Williams_Alligator_DMI_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator: 13/8/5 period SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # DMI (ADX) calculation
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = np.where(tr_sum > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 28  # Need 14*2 for ADX smoothing
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignments
        green_alignment = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])   # Bullish: jaws<teeth<lips
        red_alignment = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])    # Bearish: jaws>teeth>lips
        
        # ADX filter: trend strength
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: green alignment + strong trend
            if green_alignment and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: red alignment + strong trend
            elif red_alignment and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: alignment breaks or trend weakens
            if not green_alignment or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: alignment breaks or trend weakens
            if not red_alignment or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals