#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: Daily Donchian channel breakouts with weekly trend filter and volume
confirmation capture momentum moves. In bull markets, breakouts above upper band
with uptrend continue upward. In bear markets, breakdowns below lower band with
downtrend continue downward. Volume confirms institutional participation.
Targets 7-25 trades/year by requiring confluence of breakout, trend, and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly OHLC for Donchian calculation (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (shift by 1 week for completed bars only)
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Weekly EMA100 for trend filter
    ema100_1w = pd.Series(df_1w['close'].values).ewm(span=100, adjust=False).mean().values
    ema100_1d = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema100_1d[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns down
            if close[i] < donchian_low[i] or close[i] < ema100_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns up
            if close[i] > donchian_high[i] or close[i] > ema100_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high + volume + uptrend
            if (close[i] > donchian_high[i] and 
                vol_confirm and 
                close[i] > ema100_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low + volume + downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and 
                  close[i] < ema100_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals