#!/usr/bin/env python3
# 6h_ElderRay_Alligator_WeeklyTrend_v2
# Hypothesis: Combines Elder Ray (bull/bear power) with Williams Alligator to identify strong trends,
# filtered by weekly trend direction (EMA21). Uses 12h timeframe for Elder Ray calculation
# and 1d for Alligator to reduce noise. Designed for 6h to achieve 12-37 trades/year in both bull and bear markets.

name = "6h_ElderRay_Alligator_WeeklyTrend_v2"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 12h data for Elder Ray calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Elder Ray components on 12h
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # Align Elder Ray to 6h (wait for 12h bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator lines (SMAs)
    jaw_1d = pd.Series(close_1d).rolling(window=13, center=False).mean().values  # 13-period SMA
    teeth_1d = pd.Series(close_1d).rolling(window=8, center=False).mean().values   # 8-period SMA
    lips_1d = pd.Series(close_1d).rolling(window=5, center=False).mean().values    # 5-period SMA
    
    # Align Alligator to 6h (wait for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or \
           np.isnan(ema_21_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or \
           np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power > 0, Bear power < 0 (Alligator jaws down), price above teeth, weekly uptrend
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                close[i] > teeth_aligned[i] and close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power > 0, Bull power < 0 (Alligator jaws up), price below teeth, weekly downtrend
            elif (bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and 
                  close[i] < teeth_aligned[i] and close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull power turns negative or price drops below lips
            if bull_power_aligned[i] < 0 or close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear power turns negative or price rises above lips
            if bear_power_aligned[i] < 0 or close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals