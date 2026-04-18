#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 50-period EMA on 1d for trend filter
    if len(close_1d) >= 50:
        ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_1d = np.full_like(close_1d, np.nan)
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    vol_period = 20
    
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-vol_period:i])
    
    # Align all data to 12h timeframe
    upper_channel_12h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_12h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_12h[i]) or np.isnan(lower_channel_12h[i]) or 
            np.isnan(ema_1d_12h[i]) or np.isnan(vol_ma_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average (1d)
        vol_confirm = volume[i] > 1.3 * vol_ma_1d_12h[i]
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_1d_12h[i]
        downtrend = close[i] < ema_1d_12h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume
            if close[i] > upper_channel_12h[i] and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume
            elif close[i] < lower_channel_12h[i] and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses
            if close[i] < lower_channel_12h[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses
            if close[i] > upper_channel_12h[i] or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0