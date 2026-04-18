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
    
    # Get weekly data for indicators (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on weekly (upper and lower bands)
    upper_channel = np.full_like(close_1w, np.nan)
    lower_channel = np.full_like(close_1w, np.nan)
    
    for i in range(19, len(close_1w)):
        upper_channel[i] = np.max(high_1w[i-19:i+1])
        lower_channel[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate 50-period EMA on weekly for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all weekly data to 1d timeframe (primary)
    upper_channel_1d = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_1d = align_htf_to_ltf(prices, df_1w, lower_channel)
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1d volume spike indicator (volume > 2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_1d[i]) or np.isnan(lower_channel_1d[i]) or 
            np.isnan(ema_50_1d[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_50_1d[i]
        downtrend = close[i] < ema_50_1d[i]
        
        # Volume confirmation: require volume spike
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume spike
            if close[i] > upper_channel_1d[i] and uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume spike
            elif close[i] < lower_channel_1d[i] and downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses
            if (close[i] < lower_channel_1d[i]) or (not uptrend):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses
            if (close[i] > upper_channel_1d[i]) or (not downtrend):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0