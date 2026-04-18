#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Trend_Volume_v1
Strategy: 12h Donchian breakout with 1D trend filter and volume confirmation.
Long: Price breaks above 10-day high in uptrend with volume confirmation.
Short: Price breaks below 10-day low in downtrend with volume confirmation.
Designed for 12h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via trend filter and volatility breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 10-day high/low and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 10-day high and low (Donchian channels)
    high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (10-period)
    vol_ma_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align all daily data to 12h timeframe
    high_10_aligned = align_htf_to_ltf(prices, df_1d, high_10)
    low_10_aligned = align_htf_to_ltf(prices, df_1d, low_10)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_10_aligned[i]) or np.isnan(low_10_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > high_10_aligned[i]
        breakout_short = close[i] < low_10_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout long
            if uptrend and vol_confirm and breakout_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakout short
            elif downtrend and vol_confirm and breakout_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakout short
            if not uptrend or vol_confirm or breakout_short:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout long
            if not downtrend or vol_confirm or breakout_long:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0