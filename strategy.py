#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Donchian breakout (20) and volume confirmation, filtered by 1d EMA50 trend.
Uses breakout continuation with volume filter to avoid false breakouts. Designed to work in both bull and bear markets
by trading breakouts in the direction of the 1d trend. Aims for 15-35 trades/year on 1h (60-140 total over 4 years).
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h
    donchian_high = align_htf_to_ltf(prices, df_4h, high_max)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_min)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume, in session, and above 1d EMA50
            if close[i] > donchian_high[i] and volume_filter[i] and session_filter[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume, in session, and below 1d EMA50
            elif close[i] < donchian_low[i] and volume_filter[i] and session_filter[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_Volume_1dEMA50_Session"
timeframe = "1h"
leverage = 1.0