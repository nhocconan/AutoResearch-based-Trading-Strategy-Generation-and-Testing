#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA50) and volume confirmation.
# Enters long when price breaks above 6h Donchian upper channel with weekly bullish trend and volume > 2x MA20.
# Enters short when price breaks below 6h Donchian lower channel with weekly bearish trend and volume > 2x MA20.
# Exits when price crosses the 6h EMA20 (mean reversion within the channel).
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~12-30/year) to work in both bull and bear markets by filtering with weekly trend.

name = "6h_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
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
    
    # Get 1w data for weekly trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get 1d data for volume confirmation (to avoid intraday noise)
    df_1d = get_htf_data(prices, '1d')
    # We'll use 6h volume but filter with 1d average volume ratio for stability
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h EMA20 for exit condition
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(ema20_6h[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel with weekly bullish trend and volume spike
            if close[i] > donchian_high[i] and close[i] > ema50_1w_aligned[i] and volume[i] > (vol_ma20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower channel with weekly bearish trend and volume spike
            elif close[i] < donchian_low[i] and close[i] < ema50_1w_aligned[i] and volume[i] > (vol_ma20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 6h EMA20 (mean reversion)
            if close[i] < ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 6h EMA20 (mean reversion)
            if close[i] > ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals