#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume spike.
# Enters long when price breaks above upper Donchian channel with 1w bullish trend (close > EMA50) and volume > 2.0x MA20.
# Enters short when price breaks below lower Donchian channel with 1w bearish trend (close < EMA50) and volume > 2.0x MA20.
# Exits when price reverts to the 20-period EMA (adaptive mean reversion).
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~7-25/year) to work in both bull and bear markets by requiring strong volume confirmation and trend alignment.

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian(20) channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2.0x 20-period average (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # Exit condition: 20-period EMA of close
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian with 1w bullish trend and volume spike
            if close[i] > donchian_high[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian with 1w bearish trend and volume spike
            elif close[i] < donchian_low[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to 20-period EMA (mean reversion in range)
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to 20-period EMA (mean reversion in range)
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals