#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Trend Filter + Volume Confirmation
# Uses Elder Ray (bull/bear power) from daily data for mean reversion signals,
# weekly EMA20 for trend filter, and volume spike for confirmation
# Designed to work in both bull and bear markets by fading extremes in strong trends
# Target: 15-35 trades/year to minimize fee drift while capturing mean reversion in trends

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Get weekly data for trend filter (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Elder Ray components from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray (standard setting)
    ema13_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        multiplier = 2 / (13 + 1)
        ema13_1d[12] = np.mean(close_1d[:13])
        for i in range(13, len(close_1d)):
            ema13_1d[i] = (close_1d[i] * multiplier) + (ema13_1d[i-1] * (1 - multiplier))
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        multiplier = 2 / (20 + 1)
        ema20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema20_1w[i] = (close_1w[i] * multiplier) + (ema20_1w[i-1] * (1 - multiplier))
    
    # Calculate 4-period average volume for spike detection (6h timeframe)
    vol_ma_6h = np.full(len(prices), np.nan)
    vol_period = 4
    for i in range(vol_period, len(prices)):
        vol_ma_6h[i] = np.mean(volume[i-vol_period:i])
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(26, 20) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_6h[i] if vol_ma_6h[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Bear Power negative (weak bears) + price above weekly EMA (uptrend) + volume
            if bear_power_aligned[i] < 0 and price > ema20_1w_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Bull Power negative (weak bulls) + price below weekly EMA (downtrend) + volume
            elif bull_power_aligned[i] < 0 and price < ema20_1w_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Bear Power becomes positive (strong bears) or trend reverses
            if bear_power_aligned[i] > 0 or price < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Bull Power becomes positive (strong bulls) or trend reverses
            if bull_power_aligned[i] > 0 or price > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_WeeklyEMA20_Volume"
timeframe = "6h"
leverage = 1.0