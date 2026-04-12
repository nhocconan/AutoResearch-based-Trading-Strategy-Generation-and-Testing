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
    
    # Get 12h data for structure and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for ATR and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-day ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(atr_1d)):
        atr_1d[i] = np.nanmean(tr[i-13:i+1])
    
    # Align ATR to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    upper_12h = np.full(len(high_12h), np.nan)
    lower_12h = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        upper_12h[i] = np.max(high_12h[i-20:i])
        lower_12h[i] = np.min(low_12h[i-20:i])
    
    upper_6h = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_6h = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Calculate 12h volume moving average (20-period)
    volume_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    vol_ma_6h = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 1d trend using SMA crossover (50/200)
    sma_50_1d = np.full(len(close_1d), np.nan)
    sma_200_1d = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        sma_50_1d[i] = np.mean(close_1d[i-50:i])
    for i in range(200, len(close_1d)):
        sma_200_1d[i] = np.mean(close_1d[i-200:i])
    daily_trend = np.where(sma_50_1d > sma_200_1d, 1, np.where(sma_50_1d < sma_200_1d, -1, 0))
    daily_trend_6h = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(vol_ma_6h[i]) or np.isnan(daily_trend_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 6h volume > 1.5x 12h average
        volume_filter = volume[i] > vol_ma_6h[i] * 1.5
        
        # Only take trades in direction of daily trend
        daily_bullish = daily_trend_6h[i] == 1
        daily_bearish = daily_trend_6h[i] == -1
        
        # Entry conditions: 12h Donchian breakout with volume confirmation and daily trend alignment
        long_breakout = (close[i] > upper_6h[i]) and volume_filter and daily_bullish
        short_breakout = (close[i] < lower_6h[i]) and volume_filter and daily_bearish
        
        # Exit conditions: touch opposite Donchian level or trend reversal or ATR stop
        long_exit = (close[i] < lower_6h[i]) or (daily_trend_6h[i] == -1) or (close[i] < upper_6h[i] - 2.0 * atr_6h[i])
        short_exit = (close[i] > upper_6h[i]) or (daily_trend_6h[i] == 1) or (close[i] > lower_6h[i] + 2.0 * atr_6h[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0