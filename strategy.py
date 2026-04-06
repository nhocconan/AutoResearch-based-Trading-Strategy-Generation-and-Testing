#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Enter long when: price breaks above Donchian upper(20), 1d close > 1d EMA(50), volume > 1.5x average
# Enter short when: price breaks below Donchian lower(20), 1d close < 1d EMA(50), volume > 1.5x average
# Exit when: opposite Donchian break occurs or price crosses 1d EMA(50)
# Uses 1d trend filter to avoid counter-trend trades, volume to confirm breakout strength
# Target: 75-150 total trades over 4 years by requiring multiple confluence factors

name = "12h_donchian_1d_ema_vol_v8"
timeframe = "12h"
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
    
    # Donchian channels on 12h (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR crosses below 1d EMA(50)
            if low[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR crosses above 1d EMA(50)
            if high[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + 1d trend + volume confirmation
            if volume[i] > volume_threshold[i]:
                if high[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i]:
                    # Bullish breakout with uptrend and volume
                    signals[i] = 0.25
                    position = 1
                elif low[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i]:
                    # Bearish breakout with downtrend and volume
                    signals[i] = -0.25
                    position = -1
    
    return signals