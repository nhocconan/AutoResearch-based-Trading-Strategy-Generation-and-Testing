#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Enters long when price breaks above upper Donchian channel with 12h bullish trend (close > EMA50) and volume > 1.5x MA20.
# Enters short when price breaks below lower Donchian channel with 12h bearish trend (close < EMA50) and volume > 1.5x MA20.
# Exits when price reverts to the 12h EMA50 (mean reversion).
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~20-50/year) to work in both bull and bear markets by requiring strong volume confirmation and trend alignment.

name = "4h_Donchian_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian with 12h bullish trend and volume spike
            if close[i] > upper_channel[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian with 12h bearish trend and volume spike
            elif close[i] < lower_channel[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to 12h EMA50 (mean reversion in range)
            if close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to 12h EMA50 (mean reversion in range)
            if close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals