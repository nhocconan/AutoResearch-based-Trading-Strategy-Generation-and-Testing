#!/usr/bin/env python3
name = "12h_Donchian20_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    ema_20w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Daily Donchian(20) breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels: upper = max(high,20), lower = min(low,20)
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume filter: > 1.3x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_20w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian upper with weekly uptrend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_20w_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower with weekly downtrend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_20w_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below Donchian lower or weekly trend change
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above Donchian upper or weekly trend change
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian(20) breakout with weekly EMA(20) trend filter and volume confirmation.
# Weekly EMA(20) captures medium-term trend, reducing whipsaw in choppy markets.
# Donchian(20) breakouts capture momentum with clear entry/exit levels.
# Volume confirms institutional participation. Position size 0.25 limits drawdown.
# Target: ~15-25 trades/year to avoid fee dust while capturing significant moves.