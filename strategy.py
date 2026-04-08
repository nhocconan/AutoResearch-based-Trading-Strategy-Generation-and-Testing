#!/usr/bin/env python3
"""
4h_triple_confirmation_v1
Hypothesis: Uses 4h Donchian breakout with 1d volume confirmation and weekly trend filter.
- Long when price breaks above Donchian(20) high, 1d volume > 1.5x 20-day average, and weekly close > weekly EMA50
- Short when price breaks below Donchian(20) low, 1d volume > 1.5x 20-day average, and weekly close < weekly EMA50
- Exit when price returns to Donchian midpoint or weekly trend reverses
- Targets 20-40 trades/year to minimize fee decay
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_triple_confirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1-day volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume (20-period average)
    daily_volume = df_1d['volume'].values
    daily_volume_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_volume_ma_aligned = align_htf_to_ltf(prices, df_1d, daily_volume_ma)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(lookback, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma[i]) or np.isnan(daily_volume_ma_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to midpoint or weekly trend turns bearish
            if close[i] <= donchian_mid[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to midpoint or weekly trend turns bullish
            if close[i] >= donchian_mid[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high with volume confirmation and weekly bullish trend
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_ma[i] and
                daily_volume[i] > 1.5 * daily_volume_ma_aligned[i] and
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume confirmation and weekly bearish trend
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_ma[i] and
                  daily_volume[i] > 1.5 * daily_volume_ma_aligned[i] and
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals