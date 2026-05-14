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
    
    # Get 12h data for Donchian channel and trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 12h
    upper_channel = np.full_like(close_12h, np.nan)
    lower_channel = np.full_like(close_12h, np.nan)
    
    for i in range(19, len(close_12h)):
        upper_channel[i] = np.max(high_12h[i-19:i+1])
        lower_channel[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate 34-period EMA on 12h for trend filter
    if len(close_12h) >= 34:
        ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    else:
        ema_12h = np.full_like(close_12h, np.nan)
    
    # Calculate ATR on 12h for volatility filter
    def calculate_atr(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder smoothing for ATR
        atr = np.full_like(high, np.nan)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, 14)
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    vol_period = 20
    
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-vol_period:i])
    
    # Align all data to 12h timeframe (primary)
    upper_channel_12h = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_12h = align_htf_to_ltf(prices, df_12h, lower_channel)
    ema_12h_12h = align_htf_to_ltf(prices, df_12h, ema_12h)
    atr_12h_12h = align_htf_to_ltf(prices, df_12h, atr_12h)
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 34, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_12h[i]) or np.isnan(lower_channel_12h[i]) or 
            np.isnan(ema_12h_12h[i]) or np.isnan(atr_12h_12h[i]) or 
            np.isnan(vol_ma_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average (1d)
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_12h[i]
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_12h_12h[i]
        downtrend = close[i] < ema_12h_12h[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_12h_12h[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume
            if close[i] > upper_channel_12h[i] and uptrend and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume
            elif close[i] < lower_channel_12h[i] and downtrend and vol_confirm and vol_filter:
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

name = "12h_Donchian20_12hEMA_VolumeTrend"
timeframe = "12h"
leverage = 1.0