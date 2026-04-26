#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high AND 1-week EMA50 trend is up (close > EMA50) AND volume > 2.0x 20-period average. Enter short when price breaks below 20-period Donchian low AND 1-week EMA50 trend is down (close < EMA50) AND volume spike. Uses weekly trend filter to capture major market direction and reduce whipsaws. Designed for moderate trade frequency (19-50/year) with edge in both bull and bear markets via weekly trend alignment and volatility-based entry. Avoids SOL-only bias by requiring BTC/ETH to show similar behavior.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 4h Donchian Channel (20-period)
    # Using rolling window on 4h data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup (20), volume MA warmup (20), EMA50 warmup (50)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Donchian levels
        breakout_above = close[i] > donchian_high[i]
        breakout_below = close[i] < donchian_low[i]
        
        # 1w trend filter
        trend_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price above Donchian high + 1w uptrend + volume spike
            long_signal = breakout_above and trend_uptrend and volume_spike[i]
            
            # Short: price below Donchian low + 1w downtrend + volume spike
            short_signal = breakout_below and trend_downtrend and volume_spike[i]
            
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
            # Exit: price breaks below Donchian low OR trend change to downtrend
            if close[i] < donchian_low[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR trend change to uptrend
            if close[i] > donchian_high[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0