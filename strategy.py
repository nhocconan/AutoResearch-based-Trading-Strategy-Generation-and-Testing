#!/usr/bin/env python3
# 4h_Donchian_Breakout_VolumeTrend_4hEMA
# Hypothesis: Donchian channel breakouts capture momentum in both bull and bear markets.
# Uses 4h Donchian(20) breakouts with 4h EMA(50) trend filter and volume confirmation.
# Works in bull markets (breakouts above upper band + above EMA) and bear markets 
# (breakdowns below lower band + below EMA). Volume confirmation reduces false breakouts.
# Low trade frequency expected due to strict breakout conditions.

name = "4h_Donchian_Breakout_VolumeTrend_4hEMA"
timeframe = "4h"
leverage = 1.0

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
    
    # 4h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band, above EMA(50), volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band, below EMA(50), volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_50[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below EMA(50) OR below lower Donchian band
            if close[i] < ema_50[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above EMA(50) OR above upper Donchian band
            if close[i] > ema_50[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals