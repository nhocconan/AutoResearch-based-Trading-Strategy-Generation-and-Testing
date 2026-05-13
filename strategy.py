#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_SMA200_Trend
Hypothesis: Use daily Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to detect institutional buying/selling pressure. Go long when Bull Power > 0 and price > weekly SMA200, short when Bear Power < 0 and price < weekly SMA200. Elder Ray captures trend strength while SMA200 filters for primary trend direction, working in both bull (buy dips) and bear (sell rallies) markets. Designed for 6h timeframe to limit trades and avoid fee drag.
"""

name = "6h_ElderRay_BullBearPower_SMA200_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray to 6h timeframe (no extra delay needed for EMA-based indicator)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get weekly SMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    sma_200_1w = pd.Series(df_1w['close']).rolling(window=200, min_periods=200).mean().values
    sma_200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(sma_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power positive (buying pressure) and price above weekly SMA200
            if bull_power_aligned[i] > 0 and close[i] > sma_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power negative (selling pressure) and price below weekly SMA200
            elif bear_power_aligned[i] < 0 and close[i] < sma_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or price breaks below SMA200
            if bull_power_aligned[i] <= 0 or close[i] < sma_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive or price breaks above SMA200
            if bear_power_aligned[i] >= 0 or close[i] > sma_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals