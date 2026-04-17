#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Daily volume average for volume filter
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # Calculate 4-hour ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-hour Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-day average volume (aligned)
        volume_filter = volume[i] > (1.5 * volume_ma20_1d_aligned[i])
        
        # Trend filter: price above/below daily EMA200
        long_trend = close[i] > ema200_1d_aligned[i]
        short_trend = close[i] < ema200_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_roll[i-1]  # break above previous period's high
        short_breakout = close[i] < low_roll[i-1]   # break below previous period's low
        
        if position == 0:
            # Long: price breaks above Donchian high with trend and volume confirmation
            if long_breakout and long_trend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with trend and volume confirmation
            elif short_breakout and short_trend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend reversal
            if close[i] < low_roll[i-1] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend reversal
            if close[i] > high_roll[i-1] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_EMA200_VolumeFilter"
timeframe = "4h"
leverage = 1.0