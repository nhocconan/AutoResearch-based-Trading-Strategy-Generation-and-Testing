# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivotDir_Volume"
timeframe = "6h"
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
    
    # Donchian channels (20-period) - breakout detection
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Weekly EMA for trend filter (21-period)
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # Volume spike detection (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(weekly_ema21_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: at least 1.5x average
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above Donchian high with weekly uptrend and above weekly R3
            if (close[i] > donch_high[i] and 
                weekly_ema21_aligned[i] > weekly_ema21_aligned[i-1] and
                close[i] > r3_aligned[i] and 
                vol_condition):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with weekly downtrend and below weekly S3
            elif (close[i] < donch_low[i] and 
                  weekly_ema21_aligned[i] < weekly_ema21_aligned[i-1] and
                  close[i] < s3_aligned[i] and 
                  vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian median or weekly trend turns down
            donch_mid = (donch_high[i] + donch_low[i]) / 2
            if close[i] < donch_mid or weekly_ema21_aligned[i] < weekly_ema21_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian median or weekly trend turns up
            donch_mid = (donch_high[i] + donch_low[i]) / 2
            if close[i] > donch_mid or weekly_ema21_aligned[i] > weekly_ema21_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian(20) breakouts filtered by weekly pivot levels and trend
# - Long: Price breaks above Donchian(20) high in weekly uptrend (EMA21 rising) and above weekly R3
# - Short: Price breaks below Donchian(20) low in weekly downtrend (EMA21 falling) and below weekly S3
# - Volume confirmation (1.5x 20-period average) reduces false breakouts
# - Weekly pivot levels provide institutional support/resistance from higher timeframe
# - Weekly EMA21 trend filter ensures alignment with multi-timeframe trend
# - Exit when price returns to Donchian midpoint or weekly trend reverses
# - Position size 0.25 targets ~60-120 trades over 4 years (15-30/year) to avoid fee drag
# - Works in both bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend)
# - Novel combination: Donchian breakouts + weekly pivot direction + volume filter not recently tried on 6h
# - Uses proper MTF data loading: weekly data loaded once, aligned with proper delay
# - Designed for BTC/ETH: avoids overtrading while capturing significant moves with institutional levels