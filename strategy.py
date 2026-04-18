#!/usr/bin/env python3
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
    
    # Get 1d data for Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-period EMA on 1d for trend filter
    if len(close_1d) >= 20:
        ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema_1d = np.full_like(close_1d, np.nan)
    
    # Get 4h data for volume average
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period volume average on 4h
    vol_ma_4h = np.full_like(volume_4h, np.nan)
    vol_period = 20
    
    if len(volume_4h) >= vol_period:
        for i in range(vol_period, len(volume_4h)):
            vol_ma_4h[i] = np.mean(volume_4h[i-vol_period:i])
    
    # Align all data to 4h timeframe (primary)
    upper_channel_4h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_4h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_4h_4h = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_4h[i]) or np.isnan(lower_channel_4h[i]) or 
            np.isnan(ema_1d_4h[i]) or np.isnan(vol_ma_4h_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average (4h)
        vol_confirm = volume[i] > 1.5 * vol_ma_4h_4h[i]
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_1d_4h[i]
        downtrend = close[i] < ema_1d_4h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume
            if close[i] > upper_channel_4h[i] and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume
            elif close[i] < lower_channel_4h[i] and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA (trend change)
            if not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA (trend change)
            if not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA_VolumeTrend_v3"
timeframe = "4h"
leverage = 1.0