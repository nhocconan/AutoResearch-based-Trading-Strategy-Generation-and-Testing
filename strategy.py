#!/usr/bin/env python3
"""
6H_ElderRay_BullBearPower_1dTrend_Filter
Hypothesis: Uses Elder Ray index (Bull Power = High - EMA13, Bear Power = Low - EMA13) from 1d timeframe 
to identify bull/bear regime, then enters long when Bull Power > 0 and price > 6h EMA20, short when 
Bear Power < 0 and price < 6h EMA20. Avoids counter-trend trades by aligning with 1d trend. 
Designed for 6h timeframe with low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in both bull and bear markets by following 1d trend direction.
"""

name = "6H_ElderRay_BullBearPower_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation (EMA13 and Bull/Bear Power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align 1d indicators to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h EMA20 for trend filter
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Warmup for EMA20 and EMA13
    
    for i in range(start_idx, n):
        if np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema20_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters from 1d Elder Ray
        bull_regime = bull_power_aligned[i] > 0  # Bullish when Bull Power > 0
        bear_regime = bear_power_aligned[i] < 0  # Bearish when Bear Power < 0
        
        # 6h price relative to EMA20
        price_above_ema = close[i] > ema20_6h[i]
        price_below_ema = close[i] < ema20_6h[i]
        
        if position == 0:
            # Long entry: bull regime + price above 6h EMA20
            if bull_regime and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: bear regime + price below 6h EMA20
            elif bear_regime and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bear regime or price below 6h EMA20
            if bear_regime or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bull regime or price above 6h EMA20
            if bull_regime or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals