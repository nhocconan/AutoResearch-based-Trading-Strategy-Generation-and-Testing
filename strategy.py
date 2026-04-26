#!/usr/bin/env python3
"""
6h_ElderRay_Alligator_1dTrend_v1
Hypothesis: Combine Elder Ray (Bull/Bear Power) with Williams Alligator on 6h, filtered by 1d EMA50 trend. Elder Ray measures bull/bear power via EMA13, Alligator (JAW/TEETH/LIPS) provides trend/filter. Only trade when both agree and aligned with 1d trend. Designed for low frequency (~20-40/year) by requiring triple confluence: Elder Ray signal + Alligator alignment + 1d trend filter. Works in bull (long when bull power >0, price above Alligator, 1d uptrend) and bear (short when bear power <0, price below Alligator, 1d downtrend).
"""

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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h EMA13 for Elder Ray (Bull/Bear Power)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Williams Alligator on 6h: SMAs shifted
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    # Using EMA as proxy for SMMA (similar smoothing)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Alligator alignment: Mouth open (Lips > Teeth > Jaw) for uptrend, inverse for downtrend
    # We'll use the relation: if Lips > Jaw => bullish bias, Lips < Jaw => bearish bias
    # More precise: Mouth open up when Lips > Teeth and Teeth > Jaw
    # Mouth open down when Lips < Teeth and Teeth < Jaw
    alligator_bull = (lips > teeth) & (teeth > jaw)
    alligator_bear = (lips < teeth) & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (50), 6h EMA13 (13), Alligator shifts (max 8)
    start_idx = max(50, 13, 8)  # 50 from 1d EMA, 13 from EMA13, 8 from Jaw shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        alligator_bull_val = alligator_bull[i]
        alligator_bear_val = alligator_bear[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Bull Power > 0, Alligator bullish alignment, 1d uptrend (close > EMA50)
            long_signal = (bull_power_val > 0) and alligator_bull_val and (close_val > ema_50_1d_val)
            # Short: Bear Power < 0, Alligator bearish alignment, 1d downtrend (close < EMA50)
            short_signal = (bear_power_val < 0) and alligator_bear_val and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: any condition fails
            if not ((bull_power_val > 0) and alligator_bull_val and (close_val > ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: any condition fails
            if not ((bear_power_val < 0) and alligator_bear_val and (close_val < ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Alligator_1dTrend_v1"
timeframe = "6h"
leverage = 1.0