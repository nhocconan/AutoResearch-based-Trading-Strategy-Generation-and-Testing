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
    
    # Get 1d data for daily close and 1w data for weekly trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate 34-period EMA on weekly for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Calculate 20-period Donchian channels on daily
    upper_channel_1d = np.full_like(close_1d, np.nan)
    lower_channel_1d = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel_1d[i] = np.max(close_1d[i-19:i+1])  # using close for simplicity
        lower_channel_1d[i] = np.min(close_1d[i-19:i+1])
    
    # Calculate 20-period volume average on daily
    vol_ma_1d = np.full_like(df_1d['volume'].values, np.nan)
    vol_period = 20
    volume_1d = df_1d['volume'].values
    
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-vol_period:i])
    
    # Align all data to daily timeframe
    upper_channel_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_channel_1d)
    lower_channel_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_channel_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_1d_aligned[i]) or np.isnan(lower_channel_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume
            if close[i] > upper_channel_1d_aligned[i] and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume
            elif close[i] < lower_channel_1d_aligned[i] and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses
            if close[i] < lower_channel_1d_aligned[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses
            if close[i] > upper_channel_1d_aligned[i] or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA_VolumeTrend"
timeframe = "1d"
leverage = 1.0