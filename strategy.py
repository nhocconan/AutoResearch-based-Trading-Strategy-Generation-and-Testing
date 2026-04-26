#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and choppiness regime filter.
Enters long when price breaks above R3 with bullish 1d trend and low chop (trending market).
Enters short when price breaks below S3 with bearish 1d trend and low chop.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 75-200 total trades over 4 years.
Works in both bull and bear markets by following the 1d trend direction only, avoiding range-bound whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate prior day's Camarilla levels (using prior daily bar)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's OHLC (shifted by 1 to use completed daily bar)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Camarilla calculations
    rang = prior_high - prior_low
    camarilla_r3 = prior_close + rang * 1.1 / 4
    camarilla_s3 = prior_close - rang * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load 1d data for trend filter (EMA34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness index regime filter (14-period) - loaded once from 1d
    # Chop > 61.8 = ranging, Chop < 38.2 = trending
    # We want to trade only when market is trending (Chop < 38.2)
    hl_range_1d = df_1d['high'].values - df_1d['low'].values
    atr_14_1d = pd.Series(hl_range_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_14_1d = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need prior daily bar + EMA34 + chop)
    start_idx = max(34, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_14_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish 1d trend + low chop (trending)
        if close[i] > r3_4h[i] and close[i] > ema_34_1d_aligned[i] and chop_14_1d_aligned[i] < 38.2:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish 1d trend + low chop (trending)
        elif close[i] < s3_4h[i] and close[i] < ema_34_1d_aligned[i] and chop_14_1d_aligned[i] < 38.2:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s3_4h[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r3_4h[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_ChopFilter"
timeframe = "4h"
leverage = 1.0