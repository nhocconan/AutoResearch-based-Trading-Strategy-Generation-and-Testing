#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: Use 12h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Long when price breaks above 20-bar Donchian high, EMA50 > EMA200 on 1d, and volume > 1.5x average.
# Short when price breaks below 20-bar Donchian low, EMA50 < EMA200 on 1d, and volume > 1.5x average.
# Exit on opposite breakout or when price crosses Donchian midpoint.
# Targets 12-30 trades/year to minimize fee drag while capturing strong trends with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 and EMA200
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    uptrend_1d = ema50_1d > ema200_1d
    downtrend_1d = ema50_1d < ema200_1d
    
    # Align daily trend to 12h timeframe
    uptrend_12h = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_12h = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: 30-period average on 12h
    avg_volume = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(donchian_mid[i]) or np.isnan(uptrend_12h[i]) or \
           np.isnan(downtrend_12h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or opposite signal
            if close[i] < donchian_low[i] or \
               (close[i] < donchian_mid[i] and volume[i] > 1.5 * avg_volume[i] and downtrend_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or opposite signal
            if close[i] > donchian_high[i] or \
               (close[i] > donchian_mid[i] and volume[i] > 1.5 * avg_volume[i] and uptrend_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above Donchian high with volume and uptrend bias
            if close[i] > donchian_high[i] and volume_ok and uptrend_12h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and downtrend bias
            elif close[i] < donchian_low[i] and volume_ok and downtrend_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals