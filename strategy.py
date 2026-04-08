#!/usr/bin/env python3
# [24936] 6h_1d_elder_ray_v1
# Hypothesis: 6-hour Elder Ray (Bull Power/Bear Power) with 1-day EMA filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, price > 1-day EMA(50), and volume > 1.5x average.
# Short when Bear Power > 0, Bull Power < 0, price < 1-day EMA(50), and volume > 1.5x average.
# Exit when Bull Power <= 0 or Bear Power >= 0 (loss of momentum).
# Works in both bull and bear markets by measuring bull/bear strength relative to EMA.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for EMA and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on daily close
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        # Use pandas EMA for proper calculation
        ema_series = pd.Series(close_1d).ewm(span=50, adjust=False).mean()
        ema_50 = ema_series.values
    
    # Calculate Elder Ray components on daily data
    # Bull Power = Daily High - EMA(50)
    # Bear Power = EMA(50) - Daily Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power = np.full(len(close_1d), np.nan)
    bear_power = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not np.isnan(ema_50[i]):
            bull_power[i] = high_1d[i] - ema_50[i]
            bear_power[i] = ema_50[i] - low_1d[i]
    
    # Calculate 6-hour EMA(20) for trend context (optional filter)
    ema_20 = np.full(n, np.nan)
    if n >= 20:
        ema_series_20 = pd.Series(close).ewm(span=20, adjust=False).mean()
        ema_20 = ema_series_20.values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align daily indicators to 6-hour timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20) if n >= 20 else np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: Bull Power <= 0 (loss of bullish momentum)
            if bull_power_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: Bear Power <= 0 (loss of bearish momentum)
            if bear_power_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bull Power > 0, Bear Power < 0, price > EMA(50), volume expansion
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                price > ema_50_aligned[i] and vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power > 0, Bull Power < 0, price < EMA(50), volume expansion
            elif (bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and 
                  price < ema_50_aligned[i] and vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals