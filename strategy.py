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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-day EMA for trend filter
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 14-day RSI for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-day volume average
    vol_ma = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(20, len(volume_1d)):
            vol_ma[i] = np.mean(volume_1d[i-20:i])
    
    # Align all daily data to 1h timeframe (primary)
    upper_channel_1h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_1h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_20_1h = align_htf_to_ltf(prices, df_1d, ema_20)
    rsi_1h = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_1h = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_1h[i]) or np.isnan(lower_channel_1h[i]) or 
            np.isnan(ema_20_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(vol_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma_1h[i]
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_20_1h[i]
        downtrend = close[i] < ema_20_1h[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_extreme = (rsi_1h[i] > 30) and (rsi_1h[i] < 70)
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend, volume, and RSI not extreme
            if close[i] > upper_channel_1h[i] and uptrend and vol_confirm and rsi_not_extreme:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend, volume, and RSI not extreme
            elif close[i] < lower_channel_1h[i] and downtrend and vol_confirm and rsi_not_extreme:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses OR RSI overbought
            if (close[i] < lower_channel_1h[i]) or (not uptrend) or (rsi_1h[i] >= 70):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses OR RSI oversold
            if (close[i] > upper_channel_1h[i]) or (not downtrend) or (rsi_1h[i] <= 30):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_EMA20_RSI_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0