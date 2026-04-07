#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Donchian Breakout + Volume + 4h Trend Filter
# Hypothesis: Breakout above/below 4-hour Donchian channel with volume confirmation
# and 1-day EMA trend filter works in both bull and bear markets by trading with
# the higher timeframe trend. Uses 1h for entry timing, 4h/1d for direction.
# Target: 60-150 total trades over 4 years = 15-37/year to minimize fee drag.

name = "1h_donchian_breakout_volume_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period high and low
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, low_min_20)
    
    # 1-day EMA(20) for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_20_1h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume filter: 1h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(ema_20_1h[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian low or trend changes
            if low[i] <= donchian_low_1h[i] or close[i] < ema_20_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price reaches Donchian high or trend changes
            if high[i] >= donchian_high_1h[i] or close[i] > ema_20_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Breakout in direction of 1d EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_1h[i]:  # Uptrend
                    if high[i] > donchian_high_1h[i]:  # Break above Donchian high
                        position = 1
                        signals[i] = 0.20
                else:  # Downtrend
                    if low[i] < donchian_low_1h[i]:  # Break below Donchian low
                        position = -1
                        signals[i] = -0.20
    
    return signals