#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v4
Hypothesis: On 12h timeframe, buy when price breaks above 20-period Donchian upper band with 1d uptrend filter and volume confirmation; sell when price breaks below 20-period Donchian lower band with 1d downtrend filter and volume confirmation. Uses 1d EMA50 for trend filter and volume > 1.5x 20-period average for confirmation. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drift while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v4"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    donchian_len = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(donchian_len - 1, n):
        window_high = high[i - donchian_len + 1:i + 1]
        window_low = low[i - donchian_len + 1:i + 1]
        upper_band[i] = np.max(window_high)
        lower_band[i] = np.min(window_low)
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(donchian_len - 1, n):
        vol_avg[i] = np.mean(volume[i - donchian_len + 1:i + 1])
    
    # Load 1h data for EMA50 trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    ema_len = 50
    ema_1h = np.full(len(close_1h), np.nan)
    
    for i in range(ema_len - 1, len(close_1h)):
        ema_1h[i] = np.mean(close_1h[i - ema_len + 1:i + 1])
    
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_len - 1, n):
        # Skip if data not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_1h_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band
            if close[i] < lower_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band
            if close[i] > upper_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band + 1h EMA50 uptrend + volume
            if (close[i] > upper_band[i] and 
                close[i] > ema_1h_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band + 1h EMA50 downtrend + volume
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_1h_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals