#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot points (R1/S1) derived from daily high/low/close act as strong support/resistance.
Breakouts above daily R1 or below S1 with volume confirmation and 12h trend alignment capture momentum moves.
Exit on reversion to daily pivot point (PP) or trend reversal. Position size 0.25 targets ~25-35 trades/year to minimize fee drag.
Works in both bull (breakouts with trend) and bear (mean reversion at extremes) markets via trend filter.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot points: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    daily_pp = (h_1d + l_1d + c_1d) / 3.0
    daily_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12.0
    daily_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12.0
    
    # Align daily pivots to 4h chart (wait for daily close)
    daily_pp_aligned = align_htf_to_ltf(prices, df_1d, daily_pp)
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above daily R1 with volume confirmation and uptrend
            if (close[i] > daily_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below daily S1 with volume confirmation and downtrend
            elif (close[i] < daily_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily pivot or trend reverses
            if (close[i] < daily_pp_aligned[i]) or \
               (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily pivot or trend reverses
            if (close[i] > daily_pp_aligned[i]) or \
               (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals