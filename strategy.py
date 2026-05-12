#!/usr/bin/env python3
# 1d Weekly Donchian Breakout with Volume Confirmation and Trend Filter
# Hypothesis: Weekly Donchian breakouts capture strong trends, while volume confirms momentum
# and the daily EMA50 filter ensures alignment with intermediate-term trend. Works in both
# bull and bear markets by following the direction of weekly breakouts with proper filters.
# Designed for low trade frequency (7-25/year) to minimize fee drag.

name = "1d_WeeklyDonchianBreakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0

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
    
    # === Weekly Data for Donchian Channels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly Donchian Channels (20-period)
    weekly_donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_donchian_low)
    
    # === Daily Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_daily)
    
    # === Volume Spike (20-period on daily) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high + volume spike + above daily EMA50
            if (close[i] > donchian_high_aligned[i] and 
                vol_spike[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + volume spike + below daily EMA50
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals