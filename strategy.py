#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and session filter (08-20 UTC).
# Long when price breaks above 4h Donchian upper (20) AND volume > 1.5x 20-period average AND session 08-20 UTC.
# Short when price breaks below 4h Donchian lower (20) AND volume > 1.5x 20-period average AND session 08-20 UTC.
# Exit on close crossing 4h Donchian middle (10-period average of upper/lower).
# Uses 4h for signal direction (structure), 1h only for entry timing and session filter.
# Designed for BTC/ETH with tight entries to avoid overtrading and fee drag.

name = "1h_Donchian20_4hTrend_Volume_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels (MTF)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_ma_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).mean().values
    low_ma_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).mean().values
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0  # Middle band
    
    # Align HTF arrays to 1h timeframe (wait for completed 4h bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    
    # Volume filter: current 1h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > 4h Donchian upper AND volume spike AND session
            if close[i] > upper_20_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: price < 4h Donchian lower AND volume spike AND session
            elif close[i] < lower_20_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < 4h Donchian middle
            if close[i] < middle_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price > 4h Donchian middle
            if close[i] > middle_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals