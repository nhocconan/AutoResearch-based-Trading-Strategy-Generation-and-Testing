#!/usr/bin/env python3
"""
1d_donchian_1w_trend_volume_v1
Hypothesis: Daily Donchian channel breakout (20-period) with weekly trend filter (EMA50) and volume confirmation.
In trending markets (price above/below weekly EMA50), breakouts above upper band or below lower band signal continuation.
Volume confirmation ensures breakouts are genuine. Works in both bull and bear markets by following the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_1w_trend_volume_v1"
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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average on daily
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema50_1d[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR weekly trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR weekly trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above Donchian high in uptrend
            if (close[i] > donchian_high[i] and 
                vol_confirm and 
                close[i] > ema50_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below Donchian low in downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and 
                  close[i] < ema50_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals