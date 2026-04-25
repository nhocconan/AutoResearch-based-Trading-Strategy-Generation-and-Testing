#!/usr/bin/env python3
"""
1d_WeeklyCamarilla_H3L3_Breakout_WeeklyTrend
Hypothesis: Weekly Camarilla H3/L3 breakout on daily timeframe with weekly EMA50 trend filter. 
Uses price channels from weekly structure to capture multi-week trends in both bull and bear markets.
Weekly trend filter ensures we only trade in direction of higher timeframe momentum.
Discrete position sizing (0.25) to minimize fee drag. Targets 15-25 trades/year.
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
    
    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly Camarilla levels: H3/L3
    camarilla_h3_1w = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_l3_1w = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Align to daily timeframe (completed weekly bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50 weekly bars ~ 350 daily bars)
    start_idx = 350  # conservative warmup for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above weekly H3 + weekly uptrend
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close[i] > ema_50_1w_aligned[i])
            # Short: price closes below weekly L3 + weekly downtrend
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close[i] < ema_50_1w_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below weekly L3 OR weekly trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above weekly H3 OR weekly trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyCamarilla_H3L3_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0