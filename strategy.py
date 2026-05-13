#!/usr/bin/env python3
"""
1d_Alligator_ElderRay_Trend
Hypothesis: Williams Alligator (13,8,5 SMAs) defines trend direction, Elder Ray (bull/bear power) confirms momentum, 
and weekly trend filter avoids counter-trend trades. Works in both bull/bear by only taking trend-following entries.
Target: 15-25 trades/year per symbol.
"""

name = "1d_Alligator_ElderRay_Trend"
timeframe = "1d"
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
    
    # Williams Alligator: SMA(13), SMA(8), SMA(5) - smoothed with 8,5,3 periods
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = pd.Series(jaw).rolling(window=8, min_periods=8).mean().values  # smoothed by 8
    
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = pd.Series(teeth).rolling(window=5, min_periods=5).mean().values  # smoothed by 5
    
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = pd.Series(lips).rolling(window=3, min_periods=3).mean().values  # smoothed by 3
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1w = df_1w['close'].values > ema_20_1w
    downtrend_1w = df_1w['close'].values < ema_20_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_align = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_align = lips_val < teeth_val and teeth_val < jaw_val
        
        # Elder Ray confirmation
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        
        # Weekly trend
        uptrend_weekly = uptrend_1w_aligned[i]
        downtrend_weekly = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: Bullish Alligator + positive Bull Power + weekly uptrend
            if bullish_align and bull_power_val > 0 and uptrend_weekly:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator + negative Bear Power + weekly downtrend
            elif bearish_align and bear_power_val < 0 and downtrend_weekly:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish Alligator alignment OR Bear Power turns negative
            if bearish_align or bear_power_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish Alligator alignment OR Bull Power turns positive
            if bullish_align or bull_power_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals