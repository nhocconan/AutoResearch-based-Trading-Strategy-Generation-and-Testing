#!/usr/bin/env python3
"""
12h_Turtle_Soup_v1
Strategy: 12h Turtle Soup pattern with 1D trend filter and volume confirmation.
Long: Price retests and fails below 20-day low, then closes back above it in uptrend.
Short: Price retests and fails above 20-day high, then closes back below it in downtrend.
Designed for 12h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via trend filter and mean-reversion breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 20-day high/low and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 20-day high and low (Donchian channels)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Turtle Soup conditions
        # Long: price tested below 20-day low, then closed back above it
        soup_long = low[i] < low_20_aligned[i] and close[i] > low_20_aligned[i]
        # Short: price tested above 20-day high, then closed back below it
        soup_short = high[i] > high_20_aligned[i] and close[i] < high_20_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + soup long setup
            if uptrend and vol_confirm and soup_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + soup short setup
            elif downtrend and vol_confirm and soup_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or soup short setup
            if not uptrend or vol_confirm or soup_short:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or soup long setup
            if not downtrend or vol_confirm or soup_long:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Turtle_Soup_v1"
timeframe = "12h"
leverage = 1.0