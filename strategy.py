#!/usr/bin/env python3
"""
6h Elder Ray + Regime Filter
Uses Elder Ray (Bull/Bear Power) with EMA13 from 1d trend filter.
Long when Bull Power > 0 and Bear Power < 0 (bullish divergence) in uptrend.
Short when Bear Power > 0 and Bull Power < 0 (bearish divergence) in downtrend.
Adds volume confirmation to avoid whipsaws.
Target: 20-50 trades/year.
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
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 on 1d close for trend filter
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components on 6h data
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Determine trend from 1d EMA13
        uptrend = close[i] > ema13_1d_aligned[i]
        downtrend = close[i] < ema13_1d_aligned[i]
        
        # Long: Bull power positive AND Bear power negative (bullish divergence) in uptrend + volume
        if (bull_power[i] > 0 and bear_power[i] < 0 and uptrend and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Bear power positive AND Bull power negative (bearish divergence) in downtrend + volume
        elif (bear_power[i] > 0 and bull_power[i] < 0 and downtrend and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: when divergence disappears or trend changes
        elif i > 0 and signals[i-1] != 0:
            prev_signal = signals[i-1]
            # Exit long if bull power turns negative or bear power turns positive
            if prev_signal == 0.25 and (bull_power[i] <= 0 or bear_power[i] >= 0):
                signals[i] = 0.0
            # Exit short if bear power turns negative or bull power turns positive
            elif prev_signal == -0.25 and (bear_power[i] <= 0 or bull_power[i] >= 0):
                signals[i] = 0.0
            else:
                signals[i] = prev_signal
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_ElderRay_1dEMA13_Volume"
timeframe = "6h"
leverage = 1.0