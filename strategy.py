#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Enters long when price breaks above Donchian upper band with 1w EMA50 uptrend and volume > 1.8x MA20.
# Enters short when price breaks below Donchian lower band with 1w EMA50 downtrend and volume > 1.8x MA20.
# Exits when price reverts to the 1w EMA50.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~10-25/year) to work in both bull and bear markets.

name = "1d_Donchian_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period) from 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper with 1w uptrend and volume spike
            if close[i] > donchian_upper[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower with 1w downtrend and volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to 1w EMA50 (mean reversion in range)
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to 1w EMA50 (mean reversion in range)
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals