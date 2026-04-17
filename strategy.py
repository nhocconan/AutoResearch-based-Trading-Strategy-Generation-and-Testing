#!/usr/bin/env python3
"""
4h_1w_ChaikinMoneyFlow_Divergence_Momentum
Hypothesis: On 4h timeframe, trade weekly Chaikin Money Flow (CMF) divergences with price action to capture momentum reversals. Uses weekly CMF for institutional flow confirmation and 4h price action for timing. Designed for low trade frequency (<50/year) to minimize fee drag and work in both bull and bear markets by detecting exhaustion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_cmf(high, low, close, volume, period=20):
    """
    Calculate Chaikin Money Flow (CMF):
    CMF = Sum((Close - Low) - (High - Close)) / (High - Low) * Volume / Period
    """
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    # Replace division by zero or NaN with 0
    mfm = np.where((high - low) != 0, mfm, 0)
    # Money Flow Volume
    mfv = mfm * volume
    # CMF
    cmf = pd.Series(mfv).rolling(window=period, min_periods=period).sum() / \
          pd.Series(volume).rolling(window=period, min_periods=period).sum()
    return cmf.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Data (HTF for CMF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly CMF (20-period)
    cmf_20_1w = calculate_cmf(high_1w, low_1w, close_1w, volume_1w, 20)
    cmf_20_1w_aligned = align_htf_to_ltf(prices, df_1w, cmf_20_1w)
    
    # Weekly price change for divergence detection
    price_change_1w = pd.Series(close_1w).pct_change(periods=4).values  # 4-week change
    price_change_1w_aligned = align_htf_to_ltf(prices, df_1w, price_change_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(cmf_20_1w_aligned[i]) or 
            np.isnan(price_change_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Bearish divergence: price making higher high but CMF making lower high
            # Look for price up 2%+ over 4 weeks but CMF declining
            if (price_change_1w_aligned[i] > 0.02 and 
                cmf_20_1w_aligned[i] < cmf_20_1w_aligned[i-1] and
                cmf_20_1w_aligned[i] < 0.05):  # Weak buying pressure
                signals[i] = -0.25  # Short
                position = -1
                continue
            # Bullish divergence: price making lower low but CMF making higher low
            # Look for price down -2%+ over 4 weeks but CMF rising
            elif (price_change_1w_aligned[i] < -0.02 and 
                  cmf_20_1w_aligned[i] > cmf_20_1w_aligned[i-1] and
                  cmf_20_1w_aligned[i] > -0.05):  # Weak selling pressure
                signals[i] = 0.25  # Long
                position = 1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when CMF turns negative (selling pressure)
            if cmf_20_1w_aligned[i] < -0.05:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when CMF turns positive (buying pressure)
            if cmf_20_1w_aligned[i] > 0.05:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1w_ChaikinMoneyFlow_Divergence_Momentum"
timeframe = "4h"
leverage = 1.0