#!/usr/bin/env python3
# 4h_donchian_breakout_12h1d_trend_volume_v1
# Hypothesis: Uses 4h Donchian breakout with 12h/1d trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, 12h EMA > 1d EMA (bullish trend), and volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low, 12h EMA < 1d EMA (bearish trend), and volume > 1.5x 20-period average.
# Exit when price re-enters Donchian channel or volume drops below average.
# Designed for 20-40 trades/year to avoid fee drag. Works in bull/bear via trend-following with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 12h and 1d data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    
    # Donchian channel (20-period)
    lookback = 20
    upper_channel = np.full(len(high_4h), np.nan)
    lower_channel = np.full(len(high_4h), np.nan)
    
    for i in range(lookback, len(high_4h)):
        upper_channel[i] = np.max(high_4h[i-lookback:i])
        lower_channel[i] = np.min(low_4h[i-lookback:i])
    
    # EMA trend filters
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    vol_avg = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    # Align all indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_avg_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend alignment: 12h EMA vs 1d EMA
        bullish_trend = ema_12h_aligned[i] > ema_1d_aligned[i]
        bearish_trend = ema_12h_aligned[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        vol_surge = volume[i] > 1.5 * vol_avg_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel or volume drops
            if close[i] < upper_aligned[i] or not vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel or volume drops
            if close[i] > lower_aligned[i] or not vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high, bullish trend, volume surge
            if close[i] > upper_aligned[i] and bullish_trend and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low, bearish trend, volume surge
            elif close[i] < lower_aligned[i] and bearish_trend and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals