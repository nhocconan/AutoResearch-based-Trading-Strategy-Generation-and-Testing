#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Donchian(20) from 1d defines medium-term structure: breakout above/below indicates strong momentum
- 1w EMA50 defines the long-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.8x 20-period average) reduces false breakouts
- Designed for 1d timeframe to capture major trend moves with very low frequency (target: 7-25 trades/year)
- Uses tight volume confirmation and 1w trend filter to avoid whipsaws in ranging markets
"""

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
    
    # Calculate 1d Donchian(20) channels (based on previous 20 days)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # need 20 days + current
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels: upper = max(high of last 20 days), lower = min(low of last 20 days
    lookback = 20
    donchian_upper = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align to 1d timeframe (use previous day's channels for breakout - no look-ahead)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # need 50 weeks + current
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # need Donchian(20), 1w EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 1w EMA50 AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below 1w EMA50 AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level OR crosses 1w EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long when price < Donchian lower OR > 1w EMA50 (trend change)
                if close[i] < donchian_lower_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian upper OR < 1w EMA50 (trend change)
                if close[i] > donchian_upper_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0