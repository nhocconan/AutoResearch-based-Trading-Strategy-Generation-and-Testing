#!/usr/bin/env python3
"""
6h_Weekly_Pivot_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Weekly pivot levels (R1/S1) act as significant support/resistance on 6h timeframe.
Breakouts above weekly R1 or below weekly S1 with volume confirmation and daily trend alignment
capture momentum moves while avoiding false breakouts. Weekly pivots provide stronger levels than
daily pivots due to longer-term accumulation. Position size 0.25 targets ~15-30 trades/year to
minimize fee drag. Works in bull markets (breakouts with trend) and bear markets (mean reversion
from extreme levels via trend filter preventing counter-trend entries).
"""

name = "6h_Weekly_Pivot_R1_S1_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    h_w = df_weekly['high'].values
    l_w = df_weekly['low'].values
    c_w = df_weekly['close'].values
    
    weekly_pp = (h_w + l_w + c_w) / 3.0
    weekly_r1 = c_w + (h_w - l_w) * 1.1 / 12.0
    weekly_s1 = c_w - (h_w - l_w) * 1.1 / 12.0
    
    # Align weekly pivot levels to 6h chart (no additional delay needed)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    ema50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above weekly R1 with volume confirmation and uptrend
            if (close[i] > weekly_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_daily_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S1 with volume confirmation and downtrend
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_daily_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot point or trend reverses
            if (close[i] < weekly_pp_aligned[i]) or \
               (close[i] < ema50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot point or trend reverses
            if (close[i] > weekly_pp_aligned[i]) or \
               (close[i] > ema50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals