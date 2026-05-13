# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Trend_Volume
Hypothesis: Weekly pivot points (R1/S1) on weekly timeframe act as major support/resistance.
Breakouts above weekly R1 or below weekly S1 with volume confirmation and daily trend
alignment capture significant moves in both bull and bear markets. Exit on reversion to
weekly pivot point or trend reversal. Position size 0.25 targets 10-20 trades/year
to minimize fee drag and improve generalization across market regimes.
"""

name = "1d_Weekly_Pivot_R1_S1_Breakout_Trend_Volume"
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
    
    # Get weekly data for pivot calculation (HTF)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate Weekly Pivot Points: R1, S1, PP
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    h_weekly = df_weekly['high'].values
    l_weekly = df_weekly['low'].values
    c_weekly = df_weekly['close'].values
    
    weekly_pp = (h_weekly + l_weekly + c_weekly) / 3.0
    weekly_r1 = c_weekly + (h_weekly - l_weekly) * 1.1 / 12.0
    weekly_s1 = c_weekly - (h_weekly - l_weekly) * 1.1 / 12.0
    
    # Align weekly pivot points to daily chart
    weekly_pp_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Daily trend filter: EMA50
    ema50_daily = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Breakout above weekly R1 with volume confirmation and uptrend
            if (close[i] > weekly_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_daily[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S1 with volume confirmation and downtrend
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_daily[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot or trend reverses
            if (close[i] < weekly_pp_aligned[i]) or \
               (close[i] < ema50_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot or trend reverses
            if (close[i] > weekly_pp_aligned[i]) or \
               (close[i] > ema50_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals