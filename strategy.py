#!/usr/bin/env python3
"""
12h_Williams_Alligator_ElderRay
Hypothesis: Williams Alligator (13/8/5 SMAs) identifies trend direction and phase, Elder Ray (bull/bear power) measures momentum strength. 
Long when jaw<teeth<lips (bullish alignment) AND bear power crosses above zero with volume confirmation. 
Short when jaw>teeth>lips (bearish alignment) AND bull power crosses below zero with volume confirmation. 
Timeframe: 12H reduces trade frequency to avoid fee drag while capturing multi-day trends. Works in both bull/bear by using momentum crossovers.
"""

name = "12h_Williams_Alligator_ElderRay"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Williams Alligator: 13/8/5 period SMAs (median price)
    median_price = (prices['high'].values + prices['low'].values) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Alligator's Jaw (13-period)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Alligator's Teeth (8-period)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # Alligator's Lips (5-period)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'].values - ema13
    bear_power = prices['low'].values - ema13
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(prices['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Alligator lips (5) and Elder Ray EMA13 (13)
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] < lips[i]  # Jaw < Teeth < Lips (bullish)
        bearish_alignment = jaw[i] > teeth[i] > lips[i]  # Jaw > Teeth > Lips (bearish)
        
        # Elder Ray signals
        bull_power_cross_up = bull_power[i] > 0 and bull_power[i-1] <= 0  # Cross above zero
        bear_power_cross_down = bear_power[i] < 0 and bear_power[i-1] >= 0  # Cross below zero
        
        # Volume filter
        volume_filter = prices['volume'].values[i] > vol_ema20[i] * 1.5
        
        # 1-week trend filter
        uptrend_1w = prices['close'].values[i] > ema20_1w_aligned[i]
        downtrend_1w = prices['close'].values[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: bullish Alligator alignment + bull power crosses up + volume + 1w uptrend
            if bullish_alignment and bull_power_cross_up and volume_filter and uptrend_1w:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + bear power crosses down + volume + 1w downtrend
            elif bearish_alignment and bear_power_cross_down and volume_filter and downtrend_1w:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR bear power crosses down
            if bearish_alignment or bear_power_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR bull power crosses up
            if bullish_alignment or bull_power_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals