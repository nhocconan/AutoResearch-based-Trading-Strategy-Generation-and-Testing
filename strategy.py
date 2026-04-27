#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data for calculations (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly high and low for Donchian-like channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channel (20-period high/low)
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    if len(high_1w) >= 20:
        for i in range(20, len(high_1w)):
            donchian_high[i] = np.max(high_1w[i-20:i])
            donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Calculate weekly pivot points (using previous week's OHLC)
    prev_week_close = np.roll(high_1w, 1)  # Using high as proxy for close in roll
    prev_week_high = np.roll(high_1w, 1)
    prev_week_low = np.roll(low_1w, 1)
    # Fix first values
    prev_week_close[0] = np.nan
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    
    # Weekly pivot calculation (standard formula)
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_r1 = 2 * weekly_pivot - prev_week_low
    weekly_s1 = 2 * weekly_pivot - prev_week_high
    weekly_r2 = weekly_pivot + (prev_week_high - prev_week_low)
    weekly_s2 = weekly_pivot - (prev_week_high - prev_week_low)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        # Initialize with SMA of first 34 values
        ema_34_1d[33] = np.mean(close_1d[:34])
        # Calculate EMA for remaining values
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    
    # Align weekly indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(34, 20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(weekly_r2_aligned[i]) or np.isnan(weekly_s2_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above weekly R2 with volume and above daily EMA34
            if price > weekly_r2_aligned[i] and vol_filter and price > ema_34_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly S2 with volume and below daily EMA34
            elif price < weekly_s2_aligned[i] and vol_filter and price < ema_34_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly S1 or below daily EMA34
            if price < weekly_s1_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly R1 or above daily EMA34
            if price > weekly_r1_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R2_S2_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0