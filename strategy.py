#!/usr/bin/env python3
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
    
    # Get 1d data for trend filter and volume reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate 1d volume EMA20 for volume filter
    vol_ema_period = 20
    vol_ema_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= vol_ema_period:
        vol_ema_1d[vol_ema_period - 1] = np.mean(volume_1d[:vol_ema_period])
        for i in range(vol_ema_period, len(volume_1d)):
            vol_ema_1d[i] = (volume_1d[i] * (2 / (vol_ema_period + 1)) + 
                             vol_ema_1d[i - 1] * (1 - (2 / (vol_ema_period + 1))))
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    donch_period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(donch_period - 1, n):
        highest_high[i] = np.max(high[i - donch_period + 1:i + 1])
        lowest_low[i] = np.min(low[i - donch_period + 1:i + 1])
    
    # Calculate 4h ATR (14-period) for volatility filter
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period - 1] = np.mean(tr[1:atr_period])
        for i in range(atr_period, n):
            atr[i] = (tr[i] * (2 / (atr_period + 1)) + 
                      atr[i - 1] * (1 - (2 / (atr_period + 1))))
    
    # Align 1d indicators to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ema_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, ATR, EMA34, volume EMA
    start_idx = max(donch_period, atr_period, ema_period, vol_ema_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ema_1d_aligned[i]
        atr_now = atr[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Volatility filter: require reasonable volatility (avoid choppy markets)
        vol_filter = vol_filter and (atr_now > 0)
        
        if position == 0:
            # Long: price breaks above Donchian upper band with 1d EMA34 uptrend and volume filter
            if (price > highest_high[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower band with 1d EMA34 downtrend and volume filter
            elif (price < lowest_low[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian middle or trailing stop
            middle = (highest_high[i] + lowest_low[i]) / 2
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above Donchian middle
            middle = (highest_high[i] + lowest_low[i]) / 2
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0