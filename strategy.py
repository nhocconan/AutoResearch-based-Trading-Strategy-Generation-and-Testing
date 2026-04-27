#!/usr/bin/env python3
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
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 10-day Donchian channels for long-term trend
    donchian_high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_10)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_10)
    
    # Calculate 6-hour Donchian channels (20-period) for entry signals
    donchian_high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6-hour volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_high_6h[i]) or 
            np.isnan(donchian_low_6h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 10-day Donchian mid-point
        donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        price_above_trend = close[i] > donchian_mid
        price_below_trend = close[i] < donchian_mid
        
        # Volume filter: current volume above average
        volume_filter = vol_ma[i] > 0 and volume[i] > vol_ma[i] * 1.3
        
        # Breakout signals: price breaks 6-hour Donchian channels
        breakout_up = close[i] > donchian_high_6h[i]
        breakout_down = close[i] < donchian_low_6h[i]
        
        # Long conditions: bullish long-term trend + volume + upward breakout
        long_condition = (price_above_trend and 
                         volume_filter and 
                         breakout_up)
        
        # Short conditions: bearish long-term trend + volume + downward breakout
        short_condition = (price_below_trend and 
                          volume_filter and 
                          breakout_down)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_trend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_trend:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_DonchianTrend_Filter_Breakout_Volume"
timeframe = "6h"
leverage = 1.0