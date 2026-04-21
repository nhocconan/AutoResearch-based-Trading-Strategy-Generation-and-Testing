#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator reversal with 12h volume filter and 1d trend.
The Williams Alligator (Jaws, Teeth, Lips) identifies convergence (range) and divergence (trend).
We trade reversals when the Alligator lines cross after extended divergence, filtered by 12h volume
to avoid false signals and 1d EMA50 to align with higher timeframe trend. Designed for low trade
frequency (~15-30/year) to minimize fee drag in ranging and trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h average volume for filter
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = vol_12h / vol_ma_20_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h data
    median_price = (prices['high'].values + prices['low'].values) / 2
    # Jaws: 13-period SMMA, 8 bars ahead
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws, 8)
    # Teeth: 8-period SMMA, 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    # Lips: 5-period SMMA, 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_threshold = 1.2  # Volume must be 1.2x average
        
        if position == 0:
            # Bullish reversal: Lips cross above Teeth and Jaws, volume confirmation, uptrend
            if (lips[i-1] <= teeth[i-1] and lips[i] > teeth[i] and
                lips[i] > jaws[i] and
                vol_ratio > vol_threshold and
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Bearish reversal: Lips cross below Teeth and Jaws, volume confirmation, downtrend
            elif (lips[i-1] >= teeth[i-1] and lips[i] < teeth[i] and
                  lips[i] < jaws[i] and
                  vol_ratio > vol_threshold and
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Lips cross in opposite direction or trend reversal
            if position == 1 and (lips[i] < teeth[i] or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (lips[i] > teeth[i] or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_Reversal_12hVol_Filter_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0