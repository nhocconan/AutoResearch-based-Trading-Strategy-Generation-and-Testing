#!/usr/bin/env python3
"""
12h_Williams_Alligator_ElderRay_Trend
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) defines trend direction, Elder Ray (Bull/Bear Power) confirms momentum strength, and weekly price action filter avoids false signals. Works in trending and ranging markets by requiring alignment of multiple timeframe elements.
Target: 15-30 trades/year on 12h timeframe with strict entry conditions.
"""

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
    volume = prices['volume'].values
    
    # Williams Alligator: SMAs of median price (typical price)
    typical_price = (high + low + close) / 3
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: price vs weekly EMA21
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume filter: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + price above weekly EMA21 + volume
            if (lips[i] > teeth[i] > jaw[i] and bull_power[i] > 0 and 
                close[i] > ema21_1w_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + Bear Power < 0 + price below weekly EMA21 + volume
            elif (lips[i] < teeth[i] < jaw[i] and bear_power[i] < 0 and 
                  close[i] < ema21_1w_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks OR Bear Power becomes negative
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i] or bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks OR Bull Power becomes positive
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i] or bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_ElderRay_Trend"
timeframe = "12h"
leverage = 1.0