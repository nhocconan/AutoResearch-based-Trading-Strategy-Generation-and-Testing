#!/usr/bin/env python3
"""
1d_Donchian_Breakout_Trend_Filter
Breakout strategy using 1-day Donchian channels with 1-week trend filter.
Long when price breaks above 1d Donchian upper (20) and price > 1w EMA50 (uptrend).
Short when price breaks below 1d Donchian lower (20) and price < 1w EMA50 (downtrend).
Exit when price crosses back to 10-day SMA or trend filter fails.
Volume confirmation: current volume > 1.5x 20-day average volume.
Target: 10-25 trades/year per symbol.
"""

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
    
    # 1d Donchian channel (20-period)
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i - donchian_len + 1:i + 1])
        lower[i] = np.min(low[i - donchian_len + 1:i + 1])
    
    # 10-day SMA for exit
    sma_len = 10
    sma = np.full(n, np.nan)
    if n >= sma_len:
        sma[sma_len - 1] = np.mean(close[:sma_len])
        for i in range(sma_len, n):
            sma[i] = (close[i] + sma[i - 1] * (sma_len - 1)) / sma_len
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_ma_len = 20
    vol_ma = np.full(n, np.nan)
    if n >= vol_ma_len:
        vol_ma[vol_ma_len - 1] = np.mean(volume[:vol_ma_len])
        for i in range(vol_ma_len, n):
            vol_ma[i] = (volume[i] + vol_ma[i - 1] * (vol_ma_len - 1)) / vol_ma_len
    volume_ok = np.zeros(n, dtype=bool)
    volume_ok[:] = volume > (vol_ma * 1.5)
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_1w_len = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_1w_len:
        ema_1w[ema_1w_len - 1] = np.mean(close_1w[:ema_1w_len])
        for i in range(ema_1w_len, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_1w_len + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_1w_len + 1))))
    
    # Align 1w EMA50 to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, SMA, volume MA, and EMA1w
    start_idx = max(donchian_len - 1, sma_len - 1, vol_ma_len - 1, ema_1w_len - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(sma[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_ok[i]
        
        if position == 0:
            # Long: break above upper Donchian + volume + uptrend (price > 1w EMA50)
            if (price > upper[i] and vol_ok and price > ema_1w_aligned[i]):
                signals[i] = size
                position = 1
            # Short: break below lower Donchian + volume + downtrend (price < 1w EMA50)
            elif (price < lower[i] and vol_ok and price < ema_1w_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 10-day SMA or trend fails
            if (price < sma[i]) or (price < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 10-day SMA or trend fails
            if (price > sma[i]) or (price > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0