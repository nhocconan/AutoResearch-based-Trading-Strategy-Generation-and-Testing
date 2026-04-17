#!/usr/bin/env python3
"""
12h_Donchian20_1dTrend_VolumeBreakout_V1
Breakout above/below 20-period Donchian channel on 12h with 1d EMA34 trend filter and volume spike confirmation.
Long: Close > upper band + EMA34 > EMA144 + volume > 1.5x avg
Short: Close < lower band + EMA34 < EMA144 + volume > 1.5x avg
Exit: Close crosses middle band or trend reverses
Position size: 0.25
Designed to capture trending moves with trend alignment and volume confirmation.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 and EMA144 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema144_1d = close_series_1d.ewm(span=144, adjust=False, min_periods=144).mean().values
    
    # Align 1d EMAs to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema144_1d_aligned = align_htf_to_ltf(prices, df_1d, ema144_1d)
    
    # 20-period Donchian channel on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(144, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema144_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: 1d EMA34 > EMA144 for long, < for short
        ema34_gt_ema144 = ema34_1d_aligned[i] > ema144_1d_aligned[i]
        ema34_lt_ema144 = ema34_1d_aligned[i] < ema144_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above upper band + uptrend + volume spike
            if close[i] > donchian_high[i] and ema34_gt_ema144 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band + downtrend + volume spike
            elif close[i] < donchian_low[i] and ema34_lt_ema144 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close crosses below mid band or trend reverses
            if close[i] < donchian_mid[i] or not ema34_gt_ema144:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close crosses above mid band or trend reverses
            if close[i] > donchian_mid[i] or not ema34_lt_ema144:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dTrend_VolumeBreakout_V1"
timeframe = "12h"
leverage = 1.0