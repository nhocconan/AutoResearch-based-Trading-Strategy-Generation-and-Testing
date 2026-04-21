#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_1d
Hypothesis: Breakout above 4h Donchian(20) high with volume confirmation and 1d trend filter for long;
breakdown below 4h Donchian(20) low with volume confirmation and 1d trend filter for short.
Uses 1d EMA50 for trend filter to ensure alignment with higher timeframe momentum.
Exit when price crosses back below/above 20-period EMA on 4h to avoid whipsaw.
Designed to work in both bull and bear markets by following trend on 1d timeframe.
Target: 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 20-period high and low for Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h EMA20 for exit signal ===
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_4h[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        donchian_high = high_roll[i]
        donchian_low = low_roll[i]
        ema_trend_1d = ema_50_1d_aligned[i]
        ema_exit_4h = ema_20_4h[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + 1d uptrend + volume confirmation
            if (price_close > donchian_high and
                price_close > ema_trend_1d and
                vol_ratio_val > 1.5):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + 1d downtrend + volume confirmation
            elif (price_close < donchian_low and
                  price_close < ema_trend_1d and
                  vol_ratio_val > 1.5):
                signals[i] = -0.30
                position = -1
        
        elif position != 0:
            # Exit when price crosses back below/above 4h EMA20
            if position == 1 and price_close < ema_exit_4h:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_exit_4h:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_1d"
timeframe = "4h"
leverage = 1.0