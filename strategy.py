#!/usr/bin/env python3
"""
6h_ElderRay_ForceIndex_1dTrend_Filter
Hypothesis: Elder Ray (bull/bear power) combined with Force Index on 6h timeframe, filtered by 1d EMA trend, captures institutional momentum in both bull and bear markets. The Elder Ray identifies power shifts while Force Index confirms with volume. Trend filter prevents counter-trend trades. Target: 20-40 trades/year per symbol to minimize fee decay.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Force Index = (Close - Close_prev) * Volume
    force_index = np.diff(close, prepend=close[0]) * volume
    # Smooth Force Index with EMA(13)
    force_index_smooth = pd.Series(force_index).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(force_index_smooth[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray signals: Bull Power rising + Force Index positive = long
        # Bear Power falling + Force Index negative = short
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        force_positive = force_index_smooth[i] > 0
        force_negative = force_index_smooth[i] < 0
        
        long_signal = bull_rising and force_positive and uptrend
        short_signal = bear_falling and force_negative and downtrend
        
        # Exit conditions: opposite signal or power divergence
        long_exit = bear_falling and force_negative  # Bear power taking over
        short_exit = bull_rising and force_positive  # Bull power taking over
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_ForceIndex_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0