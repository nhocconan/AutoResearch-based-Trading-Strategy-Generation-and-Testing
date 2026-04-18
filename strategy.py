#!/usr/bin/env python3
"""
1d_1W_13WeekHigh_Low_Breakout_Volume
Hypothesis: Breakouts above 13-week high or below 13-week low with volume confirmation.
Uses weekly high/low from the prior 13 weeks (approximately 3 months) to capture
medium-term breakouts. Works in bull (breakouts to new highs) and bear (breakdowns
to new lows) markets. Volume filter reduces false breakouts. Position size 0.25.
Targets 15-25 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 70:  # need ~13 weeks + buffer
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for 13-week high/low
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 13-week high and low (prior 13 weeks, not including current week)
    # Using rolling window of 13 on weekly data
    high_13w = pd.Series(high_1w).rolling(window=13, min_periods=13).max().values
    low_13w = pd.Series(low_1w).rolling(window=13, min_periods=13).min().values
    
    # Shift by 1 to avoid look-ahead: use prior 13 weeks only
    high_13w = np.roll(high_13w, 1)
    low_13w = np.roll(low_13w, 1)
    # First value: use expanding window
    high_13w[0] = high_1w[0]
    low_13w[0] = low_1w[0]
    
    # Align weekly 13-week high/low to daily timeframe
    high_13w_aligned = align_htf_to_ltf(prices, df_1w, high_13w)
    low_13w_aligned = align_htf_to_ltf(prices, df_1w, low_13w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    vol_confirm = np.where(np.isnan(vol_confirm), False, vol_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(high_13w_aligned[i]) or np.isnan(low_13w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 13-week high with volume confirmation
            if close[i] > high_13w_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 13-week low with volume confirmation
            elif close[i] < low_13w_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below 13-week high or reverses down
            if close[i] < high_13w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above 13-week low or reverses up
            if close[i] > low_13w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_13WeekHigh_Low_Breakout_Volume"
timeframe = "1d"
leverage = 1.0