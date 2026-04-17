#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d EMA trend filter + 4h Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-bar high with 1d EMA50 > EMA200 (uptrend) and volume > 1.5x 20-bar volume average.
Short when price breaks below 20-bar low with 1d EMA50 < EMA200 (downtrend) and volume > 1.5x 20-bar volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years (19-50/year).
Donchian channels provide structural breakout levels; 1d EMA crossover filters for primary trend only; volume confirms participation.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation).
"""

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
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 1d EMA50 and EMA200 for trend
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_50_1d = ema(close_1d, 50)
    ema_200_1d = ema(close_1d, 200)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_4h, low_4h, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_4h_aligned[i]
        # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
        uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-bar high with uptrend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                uptrend and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-bar low with downtrend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-bar low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-bar high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dEMA50_200_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0