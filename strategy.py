#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian20_1dVolume_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period Donchian channels on 1d high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    period = 20
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    
    # Calculate 1d volume MA for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (volume_ma_1d * 1.5)
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema34_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Donchian breakout with volume confirmation and weekly trend filter
        bullish_breakout = close[i] > donchian_high_aligned[i] and volume_spike_aligned[i]
        bearish_breakout = close[i] < donchian_low_aligned[i] and volume_spike_aligned[i]
        
        # Weekly trend filter: only long when price above weekly EMA34, short when below
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long when bullish breakout + volume + uptrend
            if bullish_breakout and uptrend:
                signals[i] = 0.25
                position = 1
            # Short when bearish breakout + volume + downtrend
            elif bearish_breakout and downtrend:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below Donchian low or trend changes
            if close[i] < donchian_low_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above Donchian high or trend changes
            if close[i] > donchian_high_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals