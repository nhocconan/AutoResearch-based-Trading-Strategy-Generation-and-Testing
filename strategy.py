#!/usr/bin/env python3
"""
6h_TRIX_VolumeSpike_WeeklyTrend
Hypothesis: TRIX (15) momentum with volume spike confirmation and weekly trend filter.
Long when TRIX crosses above zero with volume spike and weekly close > weekly EMA20.
Short when TRIX crosses below zero with volume spike and weekly close < weekly EMA20.
Uses volume spike (2x 20-period average) to confirm institutional participation.
Designed for 6h timeframe to capture medium-term momentum in both bull and bear markets.
"""

name = "6h_TRIX_VolumeSpike_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15-period triple EMA) on 6h
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3.diff() / ema3.shift(1)) * 100  # Percentage change
    trix = trix.fillna(0).values
    
    # Weekly trend filter: weekly close > weekly EMA20 for uptrend
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = weekly_close > weekly_ema20
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    # Volume confirmation: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    volume_spike = vol_ratio > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for TRIX calculation
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(volume_spike)):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # TRIX zero crossover
        trix_cross_above = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_below = trix[i-1] >= 0 and trix[i] < 0
        
        if position == 0:
            # Long: TRIX crosses above zero + weekly uptrend + volume spike
            if trix_cross_above and weekly_uptrend_aligned[i] > 0.5 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + weekly downtrend + volume spike
            elif trix_cross_below and weekly_uptrend_aligned[i] < 0.5 and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: TRIX crosses back to zero or weekly trend reversal
            if position == 1:
                # Exit long: TRIX crosses below zero OR weekly downtrend
                if trix_cross_below or weekly_uptrend_aligned[i] < 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TRIX crosses above zero OR weekly uptrend
                if trix_cross_above or weekly_uptrend_aligned[i] > 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals