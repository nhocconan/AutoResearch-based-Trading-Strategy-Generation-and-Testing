#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v3
# Hypothesis: Donchian(20) breakout on 12h combined with 1d EMA100 trend filter and volume > 1.5x average.
# Long when price breaks above Donchian upper band with uptrend (price > 1d EMA100) and volume > 1.5x average.
# Short when price breaks below Donchian lower band with downtrend (price < 1d EMA100) and volume > 1.5x average.
# Exit when price crosses back to Donchian middle band (mean of upper/lower).
# Designed to capture strong breakouts with trend alignment in both bull and bear markets.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v3"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate Donchian channels on 12h data (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle band
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle band
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Breakout entries: Donchian upper breakout (long) and lower breakdown (short)
            if (close[i] > donchian_high[i]) and (close[i] > ema_100_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (close[i] < donchian_low[i]) and (close[i] < ema_100_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals