#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_volume_v1
Hypothesis: On the daily timeframe, buy when price breaks above the 20-day Donchian channel with weekly trend confirmation and volume expansion; sell when price breaks below the 20-day Donchian channel with weekly bearish trend and volume expansion. This captures medium-term breakouts aligned with the weekly trend, reducing false signals in choppy markets. Designed for low turnover (10-25 trades/year) to minimize fee impact while capturing sustained trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channel
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: EMA(20) on weekly close
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = close_1w > ema_20_1w
    weekly_downtrend = close_1w < ema_20_1w
    
    # Align weekly trend to daily timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_expansion = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25  # Maintain position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low or weekly trend turns down
            if close[i] < low_20[i] or weekly_downtrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high or weekly trend turns up
            if close[i] > high_20[i] or weekly_uptrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-day high with weekly uptrend and volume expansion
            if close[i] > high_20[i] and weekly_uptrend_aligned[i] > 0.5 and vol_expansion[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-day low with weekly downtrend and volume expansion
            elif close[i] < low_20[i] and weekly_downtrend_aligned[i] > 0.5 and vol_expansion[i]:
                position = -1
                signals[i] = -0.25
    
    return signals