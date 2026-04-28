#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeFilter
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with 1d uptrend (close > EMA50) and volume surge (>1.5x 20-bar MA). Enter short when price breaks below Donchian(20) low with 1d downtrend (close < EMA50) and volume surge. Exit on opposite Donchian break with volume. Trend filter avoids counter-trend trades, volume confirms institutional interest. Designed for ~20-40 trades/year to minimize fee drag in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily trend: bullish when close > EMA50, bearish when close < EMA50
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend)
    
    # Donchian channel (20-period) on 4h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(daily_uptrend_aligned[i]) or
            np.isnan(daily_downtrend_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume surge
        long_entry = close[i] > donchian_high[i] and daily_uptrend_aligned[i] and volume_surge[i]
        short_entry = close[i] < donchian_low[i] and daily_downtrend_aligned[i] and volume_surge[i]
        
        # Exit on opposite Donchian break with volume surge
        long_exit = close[i] < donchian_low[i] and volume_surge[i]
        short_exit = close[i] > donchian_high[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0