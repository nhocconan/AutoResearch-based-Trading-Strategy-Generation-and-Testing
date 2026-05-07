# NOTE: This strategy is designed to meet the 12h timeframe requirement and aims for 50-150 trades over 4 years.
# It uses a combination of 12h Donchian breakout, 1d trend filter (EMA), and volume confirmation.
# The strategy is designed to work in both bull and bear markets by using trend filters and volume confirmation.

#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume_v1
Hypothesis: Uses 12-hour Donchian channel breakouts for entry, filtered by 1-day EMA trend and volume confirmation.
This strategy aims to capture medium-term trends while minimizing false signals through trend and volume filters.
Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

name = "12h_Donchian_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12-period Donchian channels (highest high and lowest low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=12, min_periods=12).max().values
    donchian_low = low_series.rolling(window=12, min_periods=12).min().values
    
    # 1-day trend filter: EMA of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(12, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 1-day EMA with volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1-day EMA with volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls below Donchian low OR below 1-day EMA
            if close[i] < donchian_low[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above Donchian high OR above 1-day EMA
            if close[i] > donchian_high[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals