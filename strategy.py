#!/usr/bin/env python3
"""
1d_1w_RCIMultiplier_v1
Hypothesis: Uses Relative Candle Index Multiplier (RCIM) with weekly trend filter.
RCIM = (Close - Low)/(High - Low) * Volume normalized.
Long when RCIM > 0.7 and weekly close > weekly EMA20; short when RCIM < 0.3 and weekly close < weekly EMA20.
Designed for low trade frequency by requiring strong momentum and trend alignment.
Works in bull via RCIM>0.7, in bear via RCIM<0.3.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RCIMultiplier_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # RCIM calculation: (Close - Low)/(High - Low) * Volume normalized
    price_range = high_1d - low_1d
    price_range = np.where(price_range == 0, 1, price_range)  # avoid division by zero
    rcim = ((close_1d - low_1d) / price_range) * volume_1d
    
    # Normalize RCIM by its 20-period average
    rcim_series = pd.Series(rcim)
    rcim_ma = rcim_series.rolling(window=20, min_periods=20).mean().values
    rcim_normalized = rcim / np.where(rcim_ma == 0, 1, rcim_ma)
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Align RCIM to daily
    rcim_normalized_aligned = align_htf_to_ltf(prices, df_1d, rcim_normalized)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(rcim_normalized_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # RCIM conditions
        rcim_high = rcim_normalized_aligned[i] > 0.7
        rcim_low = rcim_normalized_aligned[i] < 0.3
        
        # Weekly trend filter
        weekly_uptrend = close_1w[-1] > ema20_1w_aligned[i] if len(close_1w) > 0 else False
        weekly_downtrend = close_1w[-1] < ema20_1w_aligned[i] if len(close_1w) > 0 else False
        
        # Entry conditions
        long_setup = rcim_high and weekly_uptrend
        short_setup = rcim_low and weekly_downtrend
        
        # Exit when RCIM reverses or trend fails
        exit_long = not rcim_high or not weekly_uptrend
        exit_short = not rcim_low or not weekly_downtrend
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals