#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Donchian_Breakout_1dTrend
Hypothesis: Weekly pivot points act as strong support/resistance. Price breaking above weekly R1 with volume confirmation and 1d EMA50 uptrend signals bullish continuation. Similarly, breaking below weekly S1 with volume and 1d EMA50 downtrend signals bearish continuation. Weekly pivots are calculated from prior week's OHLC, ensuring no look-ahead. Donchian(20) breakout adds momentum confirmation. Target: 15-25 trades/year. Works in bull (breakouts continue) and bear (mean reversion at pivots via opposite signals).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot points (prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation (use completed weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3.0
    camarilla_r1 = 2 * pivot - low_1w   # R1 equivalent
    camarilla_s1 = 2 * pivot - high_1w  # S1 equivalent
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) breakout for momentum confirmation (6h timeframe)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 20-period median (avoid chop)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > vol_median
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for weekly, 20 for Donchian/volume)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_median[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_conf = volume_confirm[i]
        size = fixed_size
        
        # Entry conditions: 
        # Long: price > weekly R1 AND above Donchian breakout AND volume confirm AND 1d EMA50 uptrend
        # Short: price < weekly S1 AND below Donchian breakout AND volume confirm AND 1d EMA50 downtrend
        long_entry = (close_val > r1_val) and (close_val > donch_high) and vol_conf and (close_val > ema_50_val)
        short_entry = (close_val < s1_val) and (close_val < donch_low) and vol_conf and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to weekly pivot or Donchian breakdown
            if close_val < pivot_val or close_val < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to weekly pivot or Donchian breakout
            if close_val > pivot_val or close_val > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_Donchian_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0