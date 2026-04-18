#!/usr/bin/env python3
"""
4h_InsideBarBreakout_Pullback
Hypothesis: Trade breakouts from inside bars (narrow range) on 4h, entering on pullback to the inside bar's midpoint in direction of 1d EMA(50) trend. Inside bars indicate consolidation; breakouts capture momentum. Pullback entry improves risk-reward. Works in bull/bear by following higher timeframe trend. Target ~30 trades/year to avoid fee drag.
"""

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
    
    # Get 4h data for inside bar detection
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Previous 4h bar's OHLC (completed bar)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    prev_close_4h[0] = close_4h[0]
    
    # Inside bar: current range inside previous range
    inside_bar = (high_4h <= prev_high_4h) & (low_4h >= prev_low_4h)
    
    # Inside bar midpoint
    ib_mid = (prev_high_4h + prev_low_4h) / 2.0
    
    # 1d EMA(50) trend
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align to 4h timeframe
    inside_bar_aligned = align_htf_to_ltf(prices, df_4h, inside_bar)
    ib_mid_aligned = align_htf_to_ltf(prices, df_4h, ib_mid)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ib_mid_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: inside bar breakout up + pullback to midpoint + above 1d EMA
            if (inside_bar_aligned[i] and 
                close[i] > prev_high_4h[i] and 
                close[i] <= ib_mid_aligned[i] * 1.02 and  # Allow small overshoot
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: inside bar breakout down + pullback to midpoint + below 1d EMA
            elif (inside_bar_aligned[i] and 
                  close[i] < prev_low_4h[i] and 
                  close[i] >= ib_mid_aligned[i] * 0.98 and  # Allow small undershoot
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below inside bar low or below 1d EMA
            if close[i] < prev_low_4h[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above inside bar high or above 1d EMA
            if close[i] > prev_high_4h[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_InsideBarBreakout_Pullback"
timeframe = "4h"
leverage = 1.0