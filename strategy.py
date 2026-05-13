#!/usr/bin/env python3
"""
12h_Weekly_Pivot_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Weekly pivot levels (R1/S1) on 1w timeframe act as key support/resistance.
Breakouts above R1 or below S1 with volume confirmation and daily trend alignment
capture momentum moves while avoiding false breakouts in ranging markets.
Exit on reversion to the weekly pivot point (PP) or trend reversal.
Position size 0.25 targets ~15-25 trades/year to minimize fee drag in both bull and bear markets.
"""

name = "12h_Weekly_Pivot_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pp = (h_1w + l_1w + c_1w) / 3.0
    weekly_r1 = c_1w + (h_1w - l_1w) * 1.1 / 12.0
    weekly_s1 = c_1w - (h_1w - l_1w) * 1.1 / 12.0
    
    # Align weekly pivot levels to 12h chart (wait for weekly close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above weekly R1 with volume confirmation and uptrend
            if (close[i] > weekly_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S1 with volume confirmation and downtrend
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot point or trend reverses
            if (close[i] < weekly_pp_aligned[i]) or \
               (close[i] < ema20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot point or trend reverses
            if (close[i] > weekly_pp_aligned[i]) or \
               (close[i] > ema20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals