#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 4h Donchian(30) breakout with 1d EMA100 trend filter and volume confirmation.
Breakouts aligned with daily EMA100 trend (bullish above, bearish below) tend to continue in both bull and bear markets.
Volume > 1.8x average confirms breakout strength. Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.
Target: 20-50 trades/year (80-200 over 4 years). Includes ATR-based stoploss to limit drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate EMA100 on 1d close
    close_1d = df_1d['close'].values
    ema_100 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 100:
        ema_100[99] = np.mean(close_1d[:100])  # SMA seed
        multiplier = 2 / (100 + 1)
        for i in range(100, len(close_1d)):
            ema_100[i] = (close_1d[i] * multiplier) + (ema_100[i-1] * (1 - multiplier))
    
    # Align 1d EMA100 to 4h timeframe (waits for 1d bar close)
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # Calculate 30-period Donchian channels on 4h data
    lookback = 30
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # 30-period average volume for spike detection
    vol_period = 30
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # ATR for stoploss
    atr_period = 14
    tr = np.zeros(n)
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:atr_period+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 30 for Donchian, 30 for volume, 100 for EMA100 seed
    start_idx = max(lookback, vol_period, 100)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(ema_100_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA100
        bullish = price > ema_100_aligned[i]
        bearish = price < ema_100_aligned[i]
        
        # Volume confirmation: > 1.8x average volume
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long breakout: price breaks above Donchian high in bullish trend with volume
            if bullish and price > highest_high[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below Donchian low in bearish trend with volume
            elif bearish and price < lowest_low[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns bearish or stoploss hit
            if price < lowest_low[i] or bearish or price < (entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns bullish or stoploss hit
            if price > highest_high[i] or bullish or price > (entry_price + 2.0 * atr[i]):
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

name = "4h_Donchian30_1dEMA100_Volume"
timeframe = "4h"
leverage = 1.0