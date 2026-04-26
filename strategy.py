#!/usr/bin/env python3
"""
1d_Donchian_20_Breakout_1wTrend_VolumeFilter
Hypothesis: On 1d timeframe, use Donchian(20) breakout for entries with 1w trend filter (close > 1w EMA34) and volume confirmation (>1.5x 20-period average). 1d is used for signal generation; 1w provides trend direction. This strategy targets medium-term trends in both bull and bear markets with tight entries to minimize fee drag. Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian(20) + 1w EMA34 + volume MA warmup
    start_idx = max(20, 34, 20)  # 20 for Donchian, 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_34_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + 1w uptrend + volume spike
            long_signal = (close[i] > donchian_high[i]) and trend_1w_uptrend and volume_spike[i]
            
            # Short: price breaks below Donchian low + 1w downtrend + volume spike
            short_signal = (close[i] < donchian_low[i]) and trend_1w_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Donchian low OR 1w trend turns down
            if (close[i] < donchian_low[i] or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Donchian high OR 1w trend turns up
            if (close[i] > donchian_high[i] or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian_20_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0