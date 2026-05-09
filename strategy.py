#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_EquiVolume_Trend_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume baseline
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d average volume for volume spike filter (20-day average)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_4h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 4h EquiVolume (price * volume) for breakout detection
    price_volume = close * volume
    vol_ema_20 = pd.Series(price_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h Donchian(20) breakout levels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(vol_avg_4h[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 1.5x 1d average volume (aligned)
        vol_spike = volume[i] > 1.5 * vol_avg_4h[i]
        
        # Pre-compute hour for session filter (UTC 8-20)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + 1d uptrend + session
            if (close[i] > donch_high[i] and 
                vol_spike and 
                close[i] > ema_50_4h[i] and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + 1d downtrend + session
            elif (close[i] < donch_low[i] and 
                  vol_spike and 
                  close[i] < ema_50_4h[i] and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below Donchian low (breakdown)
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above Donchian high (breakout)
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals