#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian high(20) with 1d uptrend and volume > 1.5x average.
# Short when price breaks below Donchian low(20) with 1d downtrend and volume > 1.5x average.
# Exit when price crosses opposite Donchian level or trend reverses.
# Designed for 20-30 trades/year to minimize fee drag while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    uptrend_1d = close_1d > ema50_1d
    downtrend_1d = close_1d < ema50_1d
    
    # Align daily trend to 4h
    uptrend_1d_4h = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_4h = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(uptrend_1d_4h[i]) or \
           np.isnan(downtrend_1d_4h[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or trend reverses
            if close[i] < low_20[i] or \
               (downtrend_1d_4h[i] and not uptrend_1d_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or trend reverses
            if close[i] > high_20[i] or \
               (uptrend_1d_4h[i] and not downtrend_1d_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above Donchian high with volume and 1d uptrend
            if close[i] > high_20[i] and volume_ok and uptrend_1d_4h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and 1d downtrend
            elif close[i] < low_20[i] and volume_ok and downtrend_1d_4h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals