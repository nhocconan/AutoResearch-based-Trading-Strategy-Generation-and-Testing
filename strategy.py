#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_Breakout_1dTrend_VolumeFilter_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-day high/low) on 1d data
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 12h timeframe (with 1-bar delay for completed daily bar)
    high_20_12h = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_12h = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: spike above 2.0x 20-period average on 12h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian and EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20_12h[i]) or np.isnan(low_20_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24, 12h bars)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours: 8 AM - 8 PM UTC
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price breaks above 20-day high, 1d uptrend (price > EMA34), volume breakout
            if (close[i] > high_20_12h[i] and 
                close[i] > ema_34_12h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low, 1d downtrend (price < EMA34), volume breakdown
            elif (close[i] < low_20_12h[i] and 
                  close[i] < ema_34_12h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 20-day low or trend reversal
            if close[i] < low_20_12h[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 20-day high or trend reversal
            if close[i] > high_20_12h[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals