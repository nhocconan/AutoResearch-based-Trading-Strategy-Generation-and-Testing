#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Weekly pivot levels act as strong support/resistance. Donchian(20) breakouts on 6h in direction of 1d EMA50 trend, with volume confirmation, generate high-probability trades. Weekly trend filter (price vs weekly EMA20) ensures alignment with larger timeframe momentum. Targets 12-30 trades/year by requiring confluence of weekly pivot, Donchian breakout, 1d trend, volume spike, and weekly trend alignment. Works in bull/bear via 1d trend filter and weekly trend to avoid counter-trend whipsaws.
Primary timeframe: 6h, HTF: 1d for trend/EMA50, 1w for weekly pivot and trend.
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for weekly pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points (based on prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Prior week's values (shifted by 1)
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev = np.roll(close_1w, 1)
    # First week: use same values (will be filtered by min_periods later)
    high_1w_prev[0] = high_1w[0]
    low_1w_prev[0] = low_1w[0]
    close_1w_prev[0] = close_1w[0]
    
    pivot = (high_1w_prev + low_1w_prev + close_1w_prev) / 3.0
    r1 = 2 * pivot - low_1w_prev
    s1 = 2 * pivot - high_1w_prev
    r2 = pivot + (high_1w_prev - low_1w_prev)
    s2 = pivot - (high_1w_prev - low_1w_prev)
    r3 = high_1w_prev + 2 * (pivot - low_1w_prev)
    s3 = low_1w_prev - 2 * (high_1w_prev - pivot)
    
    # Align weekly pivot levels to 6h (no extra delay needed as they're based on completed weekly candles)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate weekly EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Donchian(20) channels on 6h
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume spike: volume > 2.0x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 20 for weekly EMA, 20 for Donchian/volume median
    start_idx = max(50, 20, donchian_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        weekly_ema_val = ema_20_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above Donchian high with volume spike, 
            #       uptrend (close > EMA50_1d), above weekly EMA, and above weekly pivot (S1)
            long_entry = (close_val > highest_high[i]) and vol_spike and \
                         (close_val > ema_50_val) and (close_val > weekly_ema_val) and \
                         (close_val > s1_aligned[i])
            
            # Short: price breaks below Donchian low with volume spike,
            #        downtrend (close < EMA50_1d), below weekly EMA, and below weekly pivot (R1)
            short_entry = (close_val < lowest_low[i]) and vol_spike and \
                          (close_val < ema_50_val) and (close_val < weekly_ema_val) and \
                          (close_val < r1_aligned[i])
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price re-enters Donchian channel (below midpoint)
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if close_val < ema_50_val or close_val < weekly_ema_val or close_val < donchian_mid:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price re-enters Donchian channel (above midpoint)
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if close_val > ema_50_val or close_val > weekly_ema_val or close_val > donchian_mid:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0