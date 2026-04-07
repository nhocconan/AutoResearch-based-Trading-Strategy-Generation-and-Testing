#!/usr/bin/env python3
"""
1d_breakout_1w_trend_volume_v1
Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
In trending markets (price above/below weekly EMA), buy/sell breakouts of 20-day high/low.
Volume confirmation reduces false breakouts. Works in both bull and bear markets by
only trading in direction of weekly trend, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_breakout_1w_trend_volume_v1"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    
    # Align weekly EMA to daily timeframe
    ema20_1d = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low OR weekly trend turns down
            if close[i] < donchian_low[i] or close[i] < ema20_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high OR weekly trend turns up
            if close[i] > donchian_high[i] or close[i] > ema20_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above 20-day high in uptrend
            if (close[i] > donchian_high[i] and 
                vol_confirm and 
                close[i] > ema20_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short breakout: price breaks below 20-day low in downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and 
                  close[i] < ema20_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals