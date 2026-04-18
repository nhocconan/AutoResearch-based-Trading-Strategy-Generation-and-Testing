#!/usr/bin/env python3
"""
6h_Turtle_Soup_Reversal_With_1d_Trend_Filter
Hypothesis: Turtle Soup reversals (false breakouts of prior 6h highs/lows) combined with 1d EMA200 trend filter capture mean-reversion in chop and trend continuation in strong moves. Works in both bull/bear via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Prior 6-bar high/low for Turtle Soup setup
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    
    # 6-bar rolling max/min of prior high/low
    roll_max = pd.Series(high_shift).rolling(window=6, min_periods=6).max().values
    roll_min = pd.Series(low_shift).rolling(window=6, min_periods=6).min().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200  # Warmup for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_aligned[i]) or np.isnan(roll_max[i]) or np.isnan(roll_min[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_200_aligned[i]
        prior_high = roll_max[i]
        prior_low = roll_min[i]
        
        if position == 0:
            # Turtle Soup Long: false breakdown below prior low, then reversal up
            if low[i] < prior_low and close[i] > prior_low and close[i] > ema_trend:
                signals[i] = 0.25
                position = 1
            # Turtle Soup Short: false breakout above prior high, then reversal down
            elif high[i] > prior_high and close[i] < prior_high and close[i] < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on close below prior low or trend reversal
            if close[i] < prior_low or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on close above prior high or trend reversal
            if close[i] > prior_high or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Turtle_Soup_Reversal_With_1d_Trend_Filter"
timeframe = "6h"
leverage = 1.0