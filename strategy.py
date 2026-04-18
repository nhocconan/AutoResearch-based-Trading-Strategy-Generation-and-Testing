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
    
    # Get daily data for indicators (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily (upper and lower bands)
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 34-period EMA on daily for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period standard deviation for volatility filter
    vol_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    
    # Calculate 50-period SMA on daily for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align all daily data to 6h timeframe (primary)
    upper_channel_6h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_6h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_20_6h = align_htf_to_ltf(prices, df_1d, vol_20)
    sma_50_6h = align_htf_to_ltf(prices, df_1d, sma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 34, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_6h[i]) or np.isnan(lower_channel_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(vol_20_6h[i]) or np.isnan(sma_50_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below SMA
        uptrend = close[i] > sma_50_6h[i]
        downtrend = close[i] < sma_50_6h[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = vol_20_6h[i] > np.nanmedian(vol_20_6h[:i+1]) if i >= 20 else False
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and high volatility
            if close[i] > upper_channel_6h[i] and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and high volatility
            elif close[i] < lower_channel_6h[i] and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses
            if (close[i] < lower_channel_6h[i]) or (not uptrend):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses
            if (close[i] > upper_channel_6h[i]) or (not downtrend):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA34_VolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0