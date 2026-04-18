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
    
    # Get 12h data for indicators
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12-period Donchian channels on 12h
    upper_channel_12h = np.full_like(close_12h, np.nan)
    lower_channel_12h = np.full_like(close_12h, np.nan)
    
    for i in range(11, len(close_12h)):
        upper_channel_12h[i] = np.max(high_12h[i-11:i+1])
        lower_channel_12h[i] = np.min(low_12h[i-11:i+1])
    
    # Calculate 12-period EMA for trend filter
    ema_12 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate 14-period RSI for momentum filter
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 12-period volume average
    vol_ma_12h = np.full_like(volume_12h, np.nan)
    if len(volume_12h) >= 12:
        for i in range(12, len(volume_12h)):
            vol_ma_12h[i] = np.mean(volume_12h[i-12:i])
    
    # Align all 12h data to 4h timeframe
    upper_channel_4h = align_htf_to_ltf(prices, df_12h, upper_channel_12h)
    lower_channel_4h = align_htf_to_ltf(prices, df_12h, lower_channel_12h)
    ema_12_4h = align_htf_to_ltf(prices, df_12h, ema_12)
    rsi_4h = align_htf_to_ltf(prices, df_12h, rsi)
    vol_ma_4h = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(11, 12, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_4h[i]) or np.isnan(lower_channel_4h[i]) or 
            np.isnan(ema_12_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 12-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_4h[i]
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_12_4h[i]
        downtrend = close[i] < ema_12_4h[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_extreme = (rsi_4h[i] > 30) and (rsi_4h[i] < 70)
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend, volume, and RSI not extreme
            if close[i] > upper_channel_4h[i] and uptrend and vol_confirm and rsi_not_extreme:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend, volume, and RSI not extreme
            elif close[i] < lower_channel_4h[i] and downtrend and vol_confirm and rsi_not_extreme:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses OR RSI overbought
            if (close[i] < lower_channel_4h[i]) or (not uptrend) or (rsi_4h[i] >= 70):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses OR RSI oversold
            if (close[i] > upper_channel_4h[i]) or (not downtrend) or (rsi_4h[i] <= 30):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian12_12hEMA12_RSI_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0