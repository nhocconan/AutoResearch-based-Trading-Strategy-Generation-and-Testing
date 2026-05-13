#!/usr/bin/env python3
"""
6h_Weekly_Pivot_PriceAction
Hypothesis: Weekly pivot points (calculated from prior week's OHLC) act as strong support/resistance.
Price action around these levels (rejection or breakout with volume) provides high-probability
entries in both bull and bear markets. Trend filter from daily EMA34 avoids counter-trend trades.
Position size 0.25 targets ~15-25 trades/year to minimize fee drag.
"""

name = "6h_Weekly_Pivot_PriceAction"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from daily OHLC
    # Weekly high = max of daily highs over past 7 days
    # Weekly low = min of daily lows over past 7 days
    # Weekly close = close of most recent daily bar
    weekly_high = pd.Series(df_1d['high']).rolling(window=7, min_periods=7).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=7, min_periods=7).min().values
    weekly_close = df_1d['close'].values
    
    # Weekly pivot point and support/resistance levels
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pp - weekly_low
    weekly_s1 = 2 * weekly_pp - weekly_high
    weekly_r2 = weekly_pp + (weekly_high - weekly_low)
    weekly_s2 = weekly_pp - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h chart (no additional delay needed for pivot points)
    weekly_pp_aligned = align_ltf_to_htf(prices, df_1d, weekly_pp)
    weekly_r1_aligned = align_ltf_to_htf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_ltf_to_htf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_ltf_to_htf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_ltf_to_htf(prices, df_1d, weekly_s2)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_ltf_to_htf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and weekly uptrend
            if (close[i] > weekly_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation and weekly downtrend
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point or weekly trend reverses
            if (close[i] < weekly_pp_aligned[i]) or \
               (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point or weekly trend reverses
            if (close[i] > weekly_pp_aligned[i]) or \
               (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals