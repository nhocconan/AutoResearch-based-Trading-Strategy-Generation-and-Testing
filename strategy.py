#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h trend following using weekly Donchian breakout with 1d EMA filter and volume confirmation
# Weekly trend provides direction (works in bull/bear), 1d EMA filters counter-trend noise,
# Volume confirms institutional participation. Designed for 50-150 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donch_high_1w = np.full(len(df_1w), np.nan)
    donch_low_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        for i in range(19, len(df_1w)):
            donch_high_1w[i] = np.max(high_1w[i-19:i+1])
            donch_low_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian to 6h
    donch_high_6h = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_6h = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    
    # Load daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_6h[i]) or
            np.isnan(donch_low_6h[i]) or
            np.isnan(ema_50_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation required
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high AND above daily EMA50
            if close[i] > donch_high_6h[i] and close[i] > ema_50_6h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below weekly Donchian low AND below daily EMA50
            elif close[i] < donch_low_6h[i] and close[i] < ema_50_6h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below weekly Donchian low OR below daily EMA50
            if close[i] < donch_low_6h[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above weekly Donchian high OR above daily EMA50
            if close[i] > donch_high_6h[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Donchian_1dEMA50_Volume_Filter"
timeframe = "6h"
leverage = 1.0