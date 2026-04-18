#!/usr/bin/env python3
"""
6h Weekly Opening Gap Fill with Volume Confirmation
Trade gaps between weekly open and Friday close, expecting mean reversion to weekly open.
Works in both bull and bear markets as gaps often fill regardless of trend direction.
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
    
    # Get weekly data for gap detection
    df_1w = get_htf_data(prices, '1w')
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly gap (weekly open - previous weekly close)
    weekly_gap = open_1w - close_1w
    
    # Align weekly data to 6h timeframe
    weekly_gap_aligned = align_htf_to_ltf(prices, df_1w, weekly_gap)
    weekly_open_aligned = align_htf_to_ltf(prices, df_1w, open_1w)
    
    # Volume confirmation: 2x 50-period average (~12.5 days)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_gap_aligned[i]) or np.isnan(weekly_open_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        gap = weekly_gap_aligned[i]
        weekly_open = weekly_open_aligned[i]
        
        if position == 0:
            # Long: negative gap (week open < prev week close) + price below weekly open + volume spike
            if gap < 0 and price < weekly_open and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: positive gap (week open > prev week close) + price above weekly open + volume spike
            elif gap > 0 and price > weekly_open and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches weekly open or gap closes
            if price >= weekly_open or gap >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches weekly open or gap closes
            if price <= weekly_open or gap <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyGapFill_Volume"
timeframe = "6h"
leverage = 1.0