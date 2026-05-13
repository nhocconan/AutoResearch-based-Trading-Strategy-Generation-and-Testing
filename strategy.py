#!/usr/bin/env python3
"""
1D_Weekly_Pivot_Breakout_Trend_Volume_v2
Hypothesis: Weekly pivot points (R1/S1) derived from weekly high/low/close act as strong support/resistance.
Breakouts above weekly R1 or below S1 with volume confirmation and daily trend alignment capture momentum moves.
Exit on reversion to weekly pivot point (PP) or trend reversal. Position size 0.25 targets ~15-25 trades/year to minimize fee drag.
Works in both bull (breakouts with trend) and bear (mean reversion at extremes) markets via trend filter.
"""

name = "1D_Weekly_Pivot_Breakout_Trend_Volume_v2"
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
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pp = (h_1w + l_1w + c_1w) / 3.0
    weekly_r1 = c_1w + (h_1w - l_1w) * 1.1 / 12.0
    weekly_s1 = c_1w - (h_1w - l_1w) * 1.1 / 12.0
    
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
            # LONG: Breakout above weekly R1 with volume confirmation and uptrend
            if (close[i] > weekly_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S1 with volume confirmation and downtrend
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot or trend reverses
            if (close[i] < weekly_pp_aligned[i]) or \
               (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot or trend reverses
            if (close[i] > weekly_pp_aligned[i]) or \
               (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals