#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_Filter
Hypothesis: Elder Ray (Bull/Bear Power) signals combined with 1-week EMA trend filter to capture momentum in both bull and bear markets. 
Weekly trend filter reduces whipsaws during reversals. Targets 20-40 trades/year by requiring Elder Ray divergence and weekly trend alignment.
Works in bull markets via Bull Power strength and in bear markets via Bear Power divergence with trend filter.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1-week EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Elder Ray calculations (13-period EMA as base)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA stabilization
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1-week EMA34 slope
        # Only consider uptrend if current EMA > EMA 5 periods ago
        uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[max(i-5, start_idx)]
        downtrend = ema_34_1w_aligned[i] < ema_34_1w_aligned[max(i-5, start_idx)]
        
        # Elder Ray signals with divergence
        # Long: Bull Power rising AND above zero (bullish momentum) + weekly uptrend
        long_signal = (bull_power[i] > 0 and 
                      bull_power[i] > bull_power[i-1] and  # Rising bull power
                      uptrend)
        
        # Short: Bear Power falling AND below zero (bearish momentum) + weekly downtrend
        short_signal = (bear_power[i] < 0 and 
                       bear_power[i] < bear_power[i-1] and  # Falling bear power (more negative)
                       downtrend)
        
        # Exit when power signals reverse against position
        long_exit = (position == 1 and 
                    (bull_power[i] <= 0 or bull_power[i] < bull_power[i-1]))
        short_exit = (position == -1 and 
                     (bear_power[i] >= 0 or bear_power[i] > bear_power[i-1]))
        
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

name = "6h_ElderRay_BullBearPower_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0