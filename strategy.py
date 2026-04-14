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
    
    # Load weekly data (HTF) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 49) / 50
    
    # Align weekly EMA to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for price action and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channel (20-period)
    upper = np.full_like(high_1d, np.nan)
    lower = np.full_like(low_1d, np.nan)
    if len(high_1d) >= 20:
        for i in range(19, len(high_1d)):
            upper[i] = np.max(high_1d[i-19:i+1])
            lower[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to daily timeframe (1:1 mapping)
    upper_aligned = upper  # Already on daily timeframe
    lower_aligned = lower  # Already on daily timeframe
    
    # Calculate 20-day average volume for volume filter
    vol_avg_20 = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate ATR(14) for volatility filter and position sizing
    tr = np.zeros_like(high_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_14 = np.full_like(high_1d, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(vol_avg_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 1% of price)
        if atr_14[i] < 0.01 * close_1d[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-day average
        if vol_avg_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_1d[i] / vol_avg_20[i]
        
        # Volume threshold: require significant spike (2.0x average)
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume + weekly uptrend
            if (close_1d[i] > upper_aligned[i] and 
                volume_ratio > vol_threshold and
                close_1d[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian with volume + weekly downtrend
            elif (close_1d[i] < lower_aligned[i] and 
                  volume_ratio > vol_threshold and
                  close_1d[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below lower Donchian or weekly trend turns down
            if (close_1d[i] < lower_aligned[i] or 
                close_1d[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above upper Donchian or weekly trend turns up
            if (close_1d[i] > upper_aligned[i] or 
                close_1d[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyEMA_Volume_Filter"
timeframe = "1d"
leverage = 1.0