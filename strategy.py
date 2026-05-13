#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Reversal_Capture
Hypothesis: Weekly pivot points (PP) act as institutional support/resistance. Price often reverses from weekly PP with volume and trend alignment. This mean-reversion strategy works in both bull (pullbacks in uptrend) and bear (bounces in downtrend) by capturing reversals at the weekly pivot with volume confirmation and trend filter. Position size 0.25 targets ~10-20 trades/year to minimize fee drag.
"""

name = "1d_Weekly_Pivot_Reversal_Capture"
timeframe = "1d"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = (2*PP) - L, S1 = (2*PP) - H
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pp = (h_1w + l_1w + c_1w) / 3.0
    weekly_r1 = 2 * weekly_pp - l_1w
    weekly_s1 = 2 * weekly_pp - h_1w
    
    # Align weekly pivots to daily chart (wait for weekly close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Daily trend filter: EMA50
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Price near weekly S1 with volume confirmation and uptrend
            if (close[i] <= weekly_s1_aligned[i] * 1.02 and  # within 2% of S1
                volume_filter[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near weekly R1 with volume confirmation and downtrend
            elif (close[i] >= weekly_r1_aligned[i] * 0.98 and  # within 2% of R1
                  volume_filter[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly PP or trend reverses
            if (close[i] >= weekly_pp_aligned[i] * 0.99) or \
               (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly PP or trend reverses
            if (close[i] <= weekly_pp_aligned[i] * 1.01) or \
               (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals