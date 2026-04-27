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
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day Donchian channels
    high_20 = np.full(len(high_1d), np.nan)
    low_20 = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        high_20[i] = np.max(high_1d[i-19:i+1])
        low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 14-day ATR for stop loss
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_1d[i] = np.mean(tr[:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-day EMA for trend filter
    ema_period = 20
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume ratio (current volume / 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_ratio = np.divide(volume, vol_ma, out=np.full_like(volume, np.nan), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, ATR, EMA, and volume
    start_idx = max(19, 14, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_20_aligned[i]
        lower_channel = low_20_aligned[i]
        atr = atr_1d_aligned[i]
        ema_trend = ema_1d_aligned[i]
        vol = vol_ratio[i]
        
        # Volume filter: require above average volume
        vol_filter = vol > 1.2
        
        if position == 0:
            # Long: Price breaks above upper Donchian channel with uptrend and volume
            if price > upper_channel and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Donchian channel with downtrend and volume
            elif price < lower_channel and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to midline or trend fails
            midline = (upper_channel + lower_channel) / 2
            if price < midline or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns to midline or trend fails
            midline = (upper_channel + lower_channel) / 2
            if price > midline or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Donchian_Breakout_1D_EMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0