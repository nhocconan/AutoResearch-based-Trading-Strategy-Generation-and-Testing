#!/usr/bin/env python3
# 6h_WilliamsR_Alligator_Trend_Combo
# Hypothesis: Williams %R identifies overbought/oversold conditions while Alligator (SMAs) defines trend. Long when %R < -80 (oversold) and price > Alligator teeth (uptrend). Short when %R > -20 (overbought) and price < Alligator teeth (downtrend). Uses 1d Williams %R for higher timeframe context to avoid counter-trend trades. Works in both bull/bear by following trend with mean-reversion entries.

name = "6h_WilliamsR_Alligator_Trend_Combo"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Williams %R and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Alligator: Jaw (13-period SMA, shifted 8), Teeth (8-period SMA, shifted 5), Lips (5-period SMA, shifted 3)
    jaw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().values
    
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set shifted values to NaN for invalid periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Alligator teeth as trend filter
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Williams %R (14) and Alligator components (13,8,5 with shifts)
    start_idx = max(14, 13+8, 8+5, 5+3)  # max of lookbacks and shifts
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Alligator alignment: price > teeth = uptrend, price < teeth = downtrend
        # Using close vs teeth for trend
        price_vs_teeth = close[i] - teeth_aligned[i]
        uptrend = price_vs_teeth > 0
        downtrend = price_vs_teeth < 0
        
        if position == 0:
            # Long entry: oversold + uptrend (price above teeth)
            if oversold and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: overbought + downtrend (price below teeth)
            elif overbought and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: overbought or trend breaks (price below lips)
            if overbought or close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: oversold or trend breaks (price above lips)
            if oversold or close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals