#!/usr/bin/env python3
"""
4h_Donchian20_Volume_1dTrendFilter
Strategy: 4h Donchian breakout with volume confirmation and 1d trend filter.
Long: Close > Donchian High(20) + volume > 1.5x 20-period avg + 1d uptrend
Short: Close < Donchian Low(20) + volume > 1.5x 20-period avg + 1d downtrend
Exit: Opposite Donchian breakout or trend reversal
Position size: 0.25
Designed to capture breakouts with trend alignment and volume confirmation.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d trend filter (close > open = uptrend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    trend_1d = (df_1d['close'] > df_1d['open']).astype(float).values  # 1 for up, 0 for down
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 4h volume average aligned
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(20, n):  # warmup for Donchian
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma20_4h_aligned[i])
        
        # Trend filter: 1d bullish/bearish
        trend_up = trend_1d_aligned[i] > 0.5  # 1d close > open
        trend_down = trend_1d_aligned[i] < 0.5  # 1d close < open
        
        # Donchian breakout signals
        breakout_up = close[i] > donch_high[i-1]  # break above previous high
        breakout_down = close[i] < donch_low[i-1]  # break below previous low
        
        # Entry signals
        if position == 0:
            # Long: Donchian breakout up + volume filter + 1d uptrend
            if breakout_up and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume filter + 1d downtrend
            elif breakout_down and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakout down or 1d trend turns down
            if breakout_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout up or 1d trend turns up
            if breakout_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1dTrendFilter"
timeframe = "4h"
leverage = 1.0