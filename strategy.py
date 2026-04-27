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
    
    # Get 1d data for 12h timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-day ATR
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr_1d = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_1d[i] = np.mean(tr[:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period Donchian channels (high/low of last 20 days)
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(high_1d), np.nan)
    for i in range(19, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-19:i+1])
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-period volume SMA
    vol_sma = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_sma[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align indicators to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_sma_12h = align_htf_to_ltf(prices, df_1d, vol_sma)
    
    # Calculate 12h EMA50 for trend filter
    ema_period = 50
    ema_12h = np.full(n, np.nan)
    if n >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema_12h[i] = (close[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i-1] * (1 - (2 / (ema_period + 1))))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, Donchian, volume, EMA
    start_idx = max(19, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_12h[i]) or np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or np.isnan(vol_sma_12h[i]) or 
            np.isnan(ema_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        upper = donchian_high_12h[i]
        lower = donchian_low_12h[i]
        vol_ma = vol_sma_12h[i]
        vol_current = volume[i]
        ema_trend = ema_12h[i]
        
        if position == 0:
            # Long: Breakout above Donchian high with volume confirmation and uptrend
            if (price > upper and vol_current > vol_ma * 1.5 and price > ema_trend):
                signals[i] = size
                position = 1
            # Short: Breakdown below Donchian low with volume confirmation and downtrend
            elif (price < lower and vol_current > vol_ma * 1.5 and price < ema_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price retrace to midpoint or trend fails
            midpoint = (upper + lower) / 2
            if price < midpoint or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price retrace to midpoint or trend fails
            midpoint = (upper + lower) / 2
            if price > midpoint or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12H_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0