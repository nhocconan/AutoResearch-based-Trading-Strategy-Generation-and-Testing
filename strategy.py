#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Breakouts aligned with daily EMA50 trend (bullish above, bearish below) tend to continue in both bull and bear markets.
Volume > 2.0x average confirms breakout strength. Uses discrete position sizes (0.0, ±0.20) to minimize fee churn.
Target: 15-37 trades/year (60-150 over 4 years). Includes ATR-based stoploss to limit drawdown.
"""

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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 20-period Donchian channels on 4h data
    lookback = 20
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high_4h = np.full(len(df_4h), np.nan)
    lowest_low_4h = np.full(len(df_4h), np.nan)
    
    for i in range(lookback, len(df_4h)):
        highest_high_4h[i] = np.max(high_4h[i-lookback:i])
        lowest_low_4h[i] = np.min(low_4h[i-lookback:i])
    
    # Align Donchian levels to 1h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Calculate 20-period average volume on 4h for spike detection
    vol_4h = df_4h['volume'].values
    vol_ma_4h = np.full(len(df_4h), np.nan)
    for i in range(20, len(df_4h)):
        vol_ma_4h[i] = np.mean(vol_4h[i-20:i])
    
    # Align volume MA to 1h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # ATR for stoploss (14-period)
    tr = np.zeros(n)
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    size = 0.20  # 20% position size
    
    # Warmup: need 50 for EMA50, 20 for Donchian/volume
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_aligned[i]) or
            np.isnan(lowest_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else 0
        
        # Determine trend from 1d EMA50
        bullish = price > ema_50_aligned[i]
        bearish = price < ema_50_aligned[i]
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long breakout: price breaks above 4h Donchian high in bullish trend with volume
            if bullish and price > highest_high_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below 4h Donchian low in bearish trend with volume
            elif bearish and price < lowest_low_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low or trend turns bearish or stoploss hit
            if price < lowest_low_aligned[i] or bearish or price < (entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high or trend turns bullish or stoploss hit
            if price > highest_high_aligned[i] or bullish or price > (entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
        
        # Track entry price for stoploss calculation
        if position != 0 and signals[i] != 0:
            if position == 1 and signals[i] == size:
                entry_price = price
            elif position == -1 and signals[i] == -size:
                entry_price = price
    
    return signals

name = "1h_Donchian20_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0