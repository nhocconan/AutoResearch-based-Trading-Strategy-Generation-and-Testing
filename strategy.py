#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian(20) channels
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14)
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = np.full(len(df_4h), np.nan)
    for i in range(14, len(df_4h)):
        atr_4h[i] = np.mean(tr[i-14:i+1])
    
    # Align 4h indicators to 1h timeframe
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian(10) channels
    high_10_1d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10_1d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align 1d indicators to 1h timeframe
    high_10_1d_aligned = align_htf_to_ltf(prices, df_1d, high_10_1d)
    low_10_1d_aligned = align_htf_to_ltf(prices, df_1d, low_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(high_20_4h_aligned[i]) or np.isnan(low_20_4h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or np.isnan(high_10_1d_aligned[i]) or 
            np.isnan(low_10_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 4h ATR > 0.3 * its 20-period MA (avoid low volatility)
        atr_ma_20_4h = np.full(len(df_4h), np.nan)
        for j in range(34, len(df_4h)):  # 14 + 19 for 20-period MA
            if not np.isnan(np.mean(atr_4h[j-19:j+1])):
                atr_ma_20_4h[j] = np.mean(atr_4h[j-19:j+1])
        atr_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_20_4h)
        vol_filter = (not np.isnan(atr_ma_20_4h_aligned[i]) and 
                     atr_4h_aligned[i] > 0.3 * atr_ma_20_4h_aligned[i])
        
        # Trend filter: price above/below 4h Donchian mid
        mid_4h = (high_20_4h_aligned[i] + low_20_4h_aligned[i]) / 2
        uptrend = close[i] > mid_4h
        downtrend = close[i] < mid_4h
        
        # Entry conditions: 1h breakout of 1d Donchian in trend direction + volatility filter
        long_entry = (close[i] > high_10_1d_aligned[i]) and uptrend and vol_filter
        short_entry = (close[i] < low_10_1d_aligned[i]) and downtrend and vol_filter
        
        # Exit conditions: opposite 1h breakout of 1d Donchian or volatility drop
        long_exit = (close[i] < low_10_1d_aligned[i]) or (not vol_filter)
        short_exit = (close[i] > high_10_1d_aligned[i]) or (not vol_filter)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_donchian_trend_vol_filter_v1"
timeframe = "1h"
leverage = 1.0