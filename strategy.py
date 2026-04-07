#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation.
In trending markets (above/below 1d EMA20), breakouts of 4h Donchian(20) with volume
signal continuation. In ranging markets, avoids false breakouts via trend filter.
Works in both bull and bear by adapting to trend filter. Target: 20-50 trades/year on 4h.
"""

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
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 4h Donchian channels (20-period)
    # Calculate rolling high/low on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter
        above_ema20 = close[i] > ema20_1d_aligned[i]
        below_ema20 = close[i] < ema20_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns bearish with volume
            if close[i] <= donchian_low[i] or (below_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns bullish with volume
            if close[i] >= donchian_high[i] or (above_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Breakout with volume and trend alignment
            if close[i] > donchian_high[i] and vol_spike and above_ema20:
                # Bullish breakout with volume and uptrend
                position = 1
                signals[i] = 0.30
            elif close[i] < donchian_low[i] and vol_spike and below_ema20:
                # Bearish breakout with volume and downtrend
                position = -1
                signals[i] = -0.30
    
    return signals