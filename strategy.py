#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and daily trend filter.
# Uses 4h for signal direction (trend and breakout levels) and 1h only for entry timing to reduce trade frequency.
# Daily EMA50 filter ensures alignment with higher timeframe trend.
# Designed to work in both bull and bear markets by following daily trend and avoiding counter-trend trades.
# Target: 15-37 trades per year (60-150 over 4 years) to minimize fee drag.
name = "1h_Donchian20_4hTrend_DailyEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h trend filter: EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    high_20_4h_1h = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_1h = align_htf_to_ltf(prices, df_4h, low_20_4h)
    ema_50_4h_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_1d_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable or outside session
        if (np.isnan(high_20_4h_1h[i]) or np.isnan(low_20_4h_1h[i]) or
            np.isnan(ema_50_4h_1h[i]) or np.isnan(ema_50_1d_1h[i]) or
            np.isnan(vol_ema20[i]) or not session_mask[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper band with volume spike and above both EMAs
            if (price > high_20_4h_1h[i] and vol_spike[i] and 
                price > ema_50_4h_1h[i] and price > ema_50_1d_1h[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower band with volume spike and below both EMAs
            elif (price < low_20_4h_1h[i] and vol_spike[i] and 
                  price < ema_50_4h_1h[i] and price < ema_50_1d_1h[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian lower band
            if price < low_20_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian upper band
            if price > high_20_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals