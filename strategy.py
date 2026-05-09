#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Russell2000_Strength_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Russell 2000 proxy (BTC volatility index)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily returns and volatility (proxy for Russell 2000 strength)
    daily_ret = np.diff(df_1d['close'].values) / df_1d['close'].values[:-1]
    daily_ret = np.concatenate([[np.nan], daily_ret])  # align with df_1d index
    vol_10d = pd.Series(daily_ret).rolling(window=10, min_periods=10).std().values
    
    # Russell 2000 strength: low volatility = risk-on environment
    russell_strength = vol_10d < np.percentile(vol_10d[~np.isnan(vol_10d)], 30)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 6h data for entry signals (Donchian breakout)
    # Calculate 20-period Donchian channels on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Align all indicators to 6h timeframe
    russell_strength_6h = align_htf_to_ltf(prices, df_1d, russell_strength)
    ema20_1w_6h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 30)  # Need enough data for Donchian and weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(russell_strength_6h[i]) or np.isnan(ema20_1w_6h[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        russell_ok = russell_strength_6h[i]
        weekly_trend = ema20_1w_6h[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_ok = volume_confirm[i]
        
        if position == 0:
            # Enter long: Donchian breakout above upper channel + Russell strength + above weekly trend
            if close[i] > upper_channel and russell_ok and close[i] > weekly_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakdown below lower channel + Russell strength + below weekly trend
            elif close[i] < lower_channel and russell_ok and close[i] < weekly_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Donchian lower channel (breakdown)
            if close[i] < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian upper channel (breakout)
            if close[i] > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals