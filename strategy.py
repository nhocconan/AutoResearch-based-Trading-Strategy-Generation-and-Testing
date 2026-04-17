#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Donchian(20) breakout + volume confirmation + session filter (08-20 UTC).
Long when price breaks above 4h Donchian(20) high with volume > 1.5x 20-period volume average.
Short when price breaks below 4h Donchian(20) low with volume > 1.5x 20-period volume average.
Use 1d EMA50 filter: long only when price > 1d EMA50, short only when price < 1d EMA50.
Designed to reduce false breakouts in ranging markets and capture strong trends.
Target: 15-37 trades/year (60-150 total over 4 years) by using 4h for direction and 1h for timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper_4h, donchian_lower_4h = donchian_channel(high_4h, low_4h, 20)
    
    # Align 4h Donchian to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Get 1d data for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(donchian_upper_4h_aligned[i]) or 
            np.isnan(donchian_lower_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian(20) high with volume and above 1d EMA50
            if (close[i] > donchian_upper_4h_aligned[i] and 
                volume_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian(20) low with volume and below 1d EMA50
            elif (close[i] < donchian_lower_4h_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian(20) low
            if close[i] < donchian_lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian(20) high
            if close[i] > donchian_upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_Breakout_Volume_EMA50Filter_Session"
timeframe = "1h"
leverage = 1.0