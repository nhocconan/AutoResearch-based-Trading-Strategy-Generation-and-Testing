#!/usr/bin/env python3
name = "6h_Williams_Alligator_ElderRay_1dTrend"
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
    
    # === 1d Williams Alligator (Jaws/Teeth/Lips) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Smoothed median price (typical price smoothed with 5-period SMA)
    median_price_1d = (high_1d + low_1d + close_1d) / 3.0
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values  # Blue line
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values   # Red line
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values    # Green line
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # === 1d Elder Ray (Bull/Bear Power) ===
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 1d Trend Filter (EMA34) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        bullish_align = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Alligator alignment: Lips < Teeth < Jaw = bearish alignment
        bearish_align = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: Bullish Alligator alignment + Bull Power > 0 + price above EMA34
            if bullish_align and (bull_power_aligned[i] > 0) and (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + Bear Power < 0 + price below EMA34
            elif bearish_align and (bear_power_aligned[i] < 0) and (close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish Alligator alignment OR Bull Power <= 0
            if (not bullish_align) or (bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish Alligator alignment OR Bear Power >= 0
            if (not bearish_align) or (bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals