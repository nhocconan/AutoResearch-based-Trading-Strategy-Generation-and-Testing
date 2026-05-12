#!/usr/bin/env python3
# 1d Weekly Donchian Breakout with Volume Spike and 1w Trend Filter
# Hypothesis: On daily chart, price breaking above/below weekly Donchian channels
# indicates strong momentum. Volume surge confirms institutional participation.
# Weekly EMA200 filter ensures alignment with long-term trend, working in both
# bull (breakouts above EMA200) and bear (breakdowns below EMA200) markets.
# Designed for low trade frequency (~10-25/year) with clear entry/exit rules.

name = "1d_WeeklyDonchian_Volume_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Data for Donchian Channels and EMA200 ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly Donchian Channels (20-period)
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_200_daily = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === Daily Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or 
            np.isnan(ema_200_daily[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high + volume spike + above weekly EMA200
            if (close[i] > donchian_high_daily[i] and 
                vol_spike[i] and
                close[i] > ema_200_daily[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + volume spike + below weekly EMA200
            elif (close[i] < donchian_low_daily[i] and 
                  vol_spike[i] and
                  close[i] < ema_200_daily[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses back below weekly Donchian low (mean reversion)
            if close[i] < donchian_low_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above weekly Donchian high (mean reversion)
            if close[i] > donchian_high_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals