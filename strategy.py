#!/usr/bin/env python3
"""
12h_Monthly_Pivot_Breakout_Trend_Volume
Hypothesis: Monthly pivot points (R1/S1) derived from monthly high/low/close act as strong support/resistance on 12h charts.
Breakouts above monthly R1 or below S1 with volume confirmation and daily trend alignment capture momentum moves.
Exit on reversion to monthly pivot point (PP) or trend reversal. Position size 0.25 targets ~12-25 trades/year to minimize fee drag.
Works in both bull (breakouts with trend) and bear (mean reversion at extremes) markets via trend filter.
"""

name = "12h_Monthly_Pivot_Breakout_Trend_Volume"
timeframe = "12h"
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
    
    # Get monthly data for pivot calculation
    df_1M = get_htf_data(prices, '1M')
    
    # Calculate monthly pivot points: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_1M = df_1M['high'].values
    l_1M = df_1M['low'].values
    c_1M = df_1M['close'].values
    
    monthly_pp = (h_1M + l_1M + c_1M) / 3.0
    monthly_r1 = c_1M + (h_1M - l_1M) * 1.1 / 12.0
    monthly_s1 = c_1M - (h_1M - l_1M) * 1.1 / 12.0
    
    # Align monthly pivots to 12h chart (wait for monthly close)
    monthly_pp_aligned = align_htf_to_ltf(prices, df_1M, monthly_pp)
    monthly_r1_aligned = align_htf_to_ltf(prices, df_1M, monthly_r1)
    monthly_s1_aligned = align_htf_to_ltf(prices, df_1M, monthly_s1)
    
    # Daily trend filter: EMA50
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above monthly R1 with volume confirmation and uptrend
            if (close[i] > monthly_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below monthly S1 with volume confirmation and downtrend
            elif (close[i] < monthly_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to monthly pivot or trend reverses
            if (close[i] < monthly_pp_aligned[i]) or \
               (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to monthly pivot or trend reverses
            if (close[i] > monthly_pp_aligned[i]) or \
               (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals