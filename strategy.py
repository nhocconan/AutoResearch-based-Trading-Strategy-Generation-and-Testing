#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian breakout with 1d trend filter + volume confirmation
    # Uses 1d EMA200 for trend filter: only take breakouts in direction of 1d trend
    # Volume confirmation: volume > 2.0 * 20-period average to filter false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-37 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d Donchian channels (20-period) based on prior 1d bar
    donchian_h = np.full(len(close_1d), np.nan)
    donchian_l = np.full(len(close_1d), np.nan)
    
    for i in range(20, len(close_1d)):
        donchian_h[i] = np.max(high_1d[i-20:i])
        donchian_l[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    donchian_h_aligned = align_htf_to_ltf(prices, df_1d, donchian_h)
    donchian_l_aligned = align_htf_to_ltf(prices, df_1d, donchian_l)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(donchian_h_aligned[i]) or 
            np.isnan(donchian_l_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Donchian breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above upper Donchian in bullish trend
        if bullish_trend:
            long_entry = (close[i] > donchian_h_aligned[i]) and volume_spike[i]
        # Short breakout: price breaks below lower Donchian in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < donchian_l_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite Donchian level or trend reversal
        long_exit = bearish_trend and close[i] < donchian_l_aligned[i]
        short_exit = bullish_trend and close[i] > donchian_h_aligned[i]
        
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

name = "6h_1d_donchian_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0