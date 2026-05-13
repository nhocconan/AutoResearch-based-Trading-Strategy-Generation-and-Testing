#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Combines price channel breakout (Donchian) with higher timeframe trend (1d EMA) and volume filter
# Works in bull markets (breakouts above upper band) and bear markets (breakouts below lower band)
# Designed for low trade frequency (~25-35/year) to minimize fee drag while capturing strong trends
# Uses 20-period Donchian channels, 34-period EMA on 1d for trend filter, and volume > 20-period average

name = "4h_Donchian_EMA34_Volume"
timeframe = "4h"
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
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    # Get 1d data for EMA trend filter (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe (properly delayed for completed 1d bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Donchian
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with uptrend (price > EMA34) and volume
            if close[i] > donchian_high[i] and close[i] > ema34_1d_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with downtrend (price < EMA34) and volume
            elif close[i] < donchian_low[i] and close[i] < ema34_1d_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals