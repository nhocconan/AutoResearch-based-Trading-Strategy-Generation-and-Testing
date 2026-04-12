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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 10-period EMA on weekly close for trend
    ema10_1w = np.full(len(close_1w), np.nan)
    alpha = 2 / (10 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema10_1w[i] = close_1w[i]
        elif not np.isnan(close_1w[i]):
            ema10_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema10_1w[i-1]
        else:
            ema10_1w[i] = ema10_1w[i-1]
    
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily
    highest_20 = np.full(len(high_1d), np.nan)
    lowest_20 = np.full(len(low_1d), np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            highest_20[i] = np.max(high_1d[i-19:i+1])
            lowest_20[i] = np.min(low_1d[i-19:i+1])
    
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate 14-day ATR for volatility filter and stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = np.full(n, np.nan)
    for i in range(13, n):
        atr14[i] = np.nanmean(tr[i-13:i+1])
    
    # Volume filter: current volume > 1.3x 20-day average volume
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema10_1w_aligned[i]) or np.isnan(highest_20_aligned[i]) or
            np.isnan(lowest_20_aligned[i]) or np.isnan(atr14[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA10
        uptrend = close[i] > ema10_1w_aligned[i]
        downtrend = close[i] < ema10_1w_aligned[i]
        
        # Volume filter
        vol_filter = volume[i] > vol_ma20[i] * 1.3
        
        # Entry conditions: Donchian breakout with trend and volume
        long_breakout = (high[i] > highest_20_aligned[i]) and uptrend and vol_filter
        short_breakout = (low[i] < lowest_20_aligned[i]) and downtrend and vol_filter
        
        # Exit conditions: opposite Donchian break or volatility drop
        long_exit = (low[i] < lowest_20_aligned[i]) or (atr14[i] < np.nanmean(atr14[max(0,i-19):i+1]) * 0.6)
        short_exit = (high[i] > highest_20_aligned[i]) or (atr14[i] < np.nanmean(atr14[max(0,i-19):i+1]) * 0.6)
        
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

name = "1d_1w_donchian_ema10_breakout_vol_filter_v1"
timeframe = "1d"
leverage = 1.0