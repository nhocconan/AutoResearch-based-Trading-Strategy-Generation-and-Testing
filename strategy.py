#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day and 1-week pivot levels + volume confirmation.
# Long: Price touches 1-day or 1-week S1/S2 support + volume > 1.5x 20-period avg + price > 4h EMA50.
# Short: Price touches 1-day or 1-week R1/R2 resistance + volume > 1.5x avg + price < 4h EMA50.
# Uses daily/weekly pivots for institutional levels, volume for confirmation, EMA50 for trend filter.
# Target: 20-50 trades/year (80-200 over 4 years) to stay within fee limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for daily pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's pivot calculation
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    r1_1d = 2 * pivot_1d - low_1d[:-1]
    s1_1d = 2 * pivot_1d - high_1d[:-1]
    r2_1d = pivot_1d + (high_1d[:-1] - low_1d[:-1])
    s2_1d = pivot_1d - (high_1d[:-1] - low_1d[:-1])
    
    # 1w data for weekly pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's pivot calculation
    pivot_1w = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3
    r1_1w = 2 * pivot_1w - low_1w[:-1]
    s1_1w = 2 * pivot_1w - high_1w[:-1]
    r2_1w = pivot_1w + (high_1w[:-1] - low_1w[:-1])
    s2_1w = pivot_1w - (high_1w[:-1] - low_1w[:-1])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align pivot levels to 4h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Check if price is near pivot levels (within 0.5% tolerance)
        near_s1_1d = abs(price - s1_1d_aligned[i]) / price < 0.005
        near_s2_1d = abs(price - s2_1d_aligned[i]) / price < 0.005
        near_r1_1d = abs(price - r1_1d_aligned[i]) / price < 0.005
        near_r2_1d = abs(price - r2_1d_aligned[i]) / price < 0.005
        near_s1_1w = abs(price - s1_1w_aligned[i]) / price < 0.005
        near_s2_1w = abs(price - s2_1w_aligned[i]) / price < 0.005
        near_r1_1w = abs(price - r1_1w_aligned[i]) / price < 0.005
        near_r2_1w = abs(price - r2_1w_aligned[i]) / price < 0.005
        
        near_support = near_s1_1d or near_s2_1d or near_s1_1w or near_s2_1w
        near_resistance = near_r1_1d or near_r2_1d or near_r1_1w or near_r2_1w
        
        if position == 0:
            # Long: near support + volume confirmation + above EMA50
            if near_support and volume_confirm and price > ema_trend:
                position = 1
                signals[i] = position_size
            # Short: near resistance + volume confirmation + below EMA50
            elif near_resistance and volume_confirm and price < ema_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price goes below EMA50 or hits resistance
            if price < ema_trend or near_resistance:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price goes above EMA50 or hits support
            if price > ema_trend or near_support:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_1w_Pivot_Volume_EMA"
timeframe = "4h"
leverage = 1.0