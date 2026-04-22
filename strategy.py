# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_WeeklyPivot_S1R1_Bounce
Hypothesis: Price tends to bounce off weekly pivot support/resistance levels (S1/R1) during ranging markets.
Uses weekly pivot levels as dynamic support/resistance with 60-minute momentum confirmation.
Works in both bull and bear markets by fading extremes at key weekly levels.
Target: 20-40 trades/year with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2*P - H, R1 = 2*P - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    s1 = 2 * pivot - high_1w  # Support 1
    r1 = 2 * pivot - low_1w   # Resistance 1
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # 60-minute momentum: ROC(60) - measures momentum over 5 periods (5*60min = 5h)
    close = prices['close'].values
    roc_60 = np.zeros_like(close)
    for i in range(5, len(close)):
        if close[i-5] != 0:
            roc_60[i] = ((close[i] - close[i-5]) / close[i-5]) * 100
    
    # Volume filter: current volume > 1.3x 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(roc_60[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        roc_val = roc_60[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Volume filter
        vol_filter = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long: price near S1 support with bullish momentum
            near_s1 = abs(price - s1_val) / s1_val < 0.005  # Within 0.5% of S1
            bullish_momentum = roc_val > 15  # Strong upward momentum
            
            if near_s1 and bullish_momentum and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # Short: price near R1 resistance with bearish momentum
            elif abs(price - r1_val) / r1_val < 0.005 and roc_val < -15 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches pivot or momentum fades
                if price >= pivot_val or roc_val < 5:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches pivot or momentum fades
                if price <= pivot_val or roc_val > -5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_S1R1_Bounce"
timeframe = "6h"
leverage = 1.0