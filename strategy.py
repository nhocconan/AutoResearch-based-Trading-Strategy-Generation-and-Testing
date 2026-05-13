#!/usr/bin/env python3
name = "12h_DonchianBreakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1D data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1D EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D EMA50 to 12H timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12H Donchian channels (20-period)
    donchian_window = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_window - 1, n):
        upper[i] = np.max(high[i-donchian_window+1:i+1])
        lower[i] = np.min(low[i-donchian_window+1:i+1])
    
    # Calculate 12H volume moving average (20-period)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1D EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + uptrend + volume spike
            if close[i] > upper[i] and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + downtrend + volume spike
            elif close[i] < lower[i] and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or trend reverses
            if close[i] < lower[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or trend reverses
            if close[i] > upper[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals