#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d/1w trend filter and volume confirmation
    # Donchian channels provide clear breakout levels based on price extremes
    # 1d EMA50 + 1w EMA200 filter for multi-timeframe trend alignment to avoid counter-trend whipsaws
    # Volume spike >1.8x 20-period average confirms institutional participation
    # Target: 12-37 trades/year (50-150 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 1w data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    upper_channel = np.full(len(high_6h), np.nan)
    lower_channel = np.full(len(low_6h), np.nan)
    
    for i in range(20, len(high_6h)):
        upper_channel[i] = np.max(high_6h[i-20:i])
        lower_channel[i] = np.min(low_6h[i-20:i])
    
    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w EMA200 for long-term trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 6h volume for confirmation (>1.8x 20-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (1.8 * vol_ma_6h)
    
    # Align all indicators to LTF (6h)
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_channel)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_channel)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_6h_aligned[i]) or np.isnan(lower_6h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_6h_aligned[i]
        short_breakout = close[i] < lower_6h_aligned[i]
        
        # Multi-timeframe trend filter (1d EMA50 + 1w EMA200)
        bullish_trend = (close[i] > ema50_1d_aligned[i]) and (close[i] > ema200_1w_aligned[i])
        bearish_trend = (close[i] < ema50_1d_aligned[i]) and (close[i] < ema200_1w_aligned[i])
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_6h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_6h[i]
        
        # Exit logic: price retests Donchian channels or trend reversal
        long_exit = (close[i] <= upper_6h_aligned[i] * 1.001) or not bullish_trend  # Retest upper or trend change
        short_exit = (close[i] >= lower_6h_aligned[i] * 0.999) or not bearish_trend  # Retest lower or trend change
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "6h_1d_1w_donchian_breakout_ema50_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0