#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with Daily Trend Filter and Volume Confirmation
# - Long when price breaks above 20-period Donchian high + price above 1d EMA50 + volume > 1.5x average
# - Short when price breaks below 20-period Donchian low + price below 1d EMA50 + volume > 1.5x average
# - Exit when price crosses the 10-period Donchian midpoint (mean reversion within the channel)
# - Uses 12h timeframe for lower frequency to reduce fee drag
# - Daily EMA50 filter ensures alignment with higher timeframe trend
# - Volume confirmation filters out weak breakouts
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_20 + low_20) / 2
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = volume[i] > 1.5 * volume_ma[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 1d EMA50 + volume confirmation
            if close[i] > high_20[i] and close[i] > ema_50_1d_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below 1d EMA50 + volume confirmation
            elif close[i] < low_20[i] and close[i] < ema_50_1d_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian middle
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian middle
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0