#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and 1d ADX regime filter
# Donchian(20) breakout captures trend continuation with volume confirmation
# 1d ADX > 25 filters for trending markets only to avoid whipsaws in ranging conditions
# Position size: 0.25 (25%) to manage drawdown during 2022-like crashes
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# Works in bull markets via trend following and avoids false signals in bear/ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data once for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data once for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for regime detection
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to main timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_12h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_12h, lowest_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume moving average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] > 25:
            # Long breakout: price breaks above 20-period high with volume confirmation
            if (close[i] > highest_high_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and position <= 0):
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below 20-period low with volume confirmation
            elif (close[i] < lowest_low_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and position >= 0):
                position = -1
                signals[i] = -position_size
            # Exit when price crosses back through the midpoint
            elif position == 1 and close[i] < (highest_high_aligned[i] + lowest_low_aligned[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > (highest_high_aligned[i] + lowest_low_aligned[i]) / 2:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Volume_ADX_Trend"
timeframe = "12h"
leverage = 1.0