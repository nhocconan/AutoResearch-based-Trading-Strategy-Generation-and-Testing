#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly close for trend filter
    weekly_close = df_1w['close'].values
    # Daily high/low for Donchian channels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Weekly EMA20 for trend filter (lagging indicator, needs extra delay)
    ema_20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Daily Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    # Daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = daily_volume / (vol_ma_20 + 1e-10)
    
    # Align HTF indicators to 12h timeframe
    # Weekly EMA20 needs extra delay as it's a lagging trend filter
    ema_20_1w_12h = align_htf_to_ltf(prices, df_1w, ema_20_1w, additional_delay_bars=1)
    # Daily Donchian channels (breakout signals need only completed bar delay)
    highest_20_12h = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_12h = align_htf_to_ltf(prices, df_1d, lowest_20)
    # Daily volume ratio (use completed bar)
    volume_ratio_12h = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    signals = np.zeros(n)
    
    # Precompute hour filter for UTC 8-20 (active session)
    hours = prices.index.hour
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_12h[i]) or np.isnan(highest_20_12h[i]) or 
            np.isnan(lowest_20_12h[i]) or np.isnan(volume_ratio_12h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: trade only during active UTC hours (8-20)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA20
        # 2. Daily Donchian breakout: price breaks 20-day high/low
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: break above 20-day high in weekly uptrend
        if (close[i] > ema_20_1w_12h[i] and      # Weekly uptrend filter
            close[i] > highest_20_12h[i] and     # Daily Donchian breakout
            volume_ratio_12h[i] > 1.5):          # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: break below 20-day low in weekly downtrend
        elif (close[i] < ema_20_1w_12h[i] and    # Weekly downtrend filter
              close[i] < lowest_20_12h[i] and    # Daily Donchian breakdown
              volume_ratio_12h[i] > 1.5):        # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WeeklyEMA20_Donchian20_Breakout_Volume"
timeframe = "12h"
leverage = 1.0