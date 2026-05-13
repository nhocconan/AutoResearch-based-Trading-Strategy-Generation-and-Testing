#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND price > 1d EMA34 AND volume > 1.5x average.
# Short when price breaks below 20-period Donchian low AND price < 1d EMA34 AND volume > 1.5x average.
# Exit when price crosses the opposite Donchian level (e.g., long exit on break below Donchian low).
# Uses 4h timeframe for lower frequency, Donchian for structure, 1d EMA for trend filter, volume for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies.

name = "4h_Donchian20_1dTrend_Volume_v1"
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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian(20) on 4h high/low
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = volume_4h > (1.5 * vol_ma_4h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > 1d EMA34 AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema34_1d_aligned[i] and volume_filter_4h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND price < 1d EMA34 AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema34_1d_aligned[i] and volume_filter_4h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (opposite breakout)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (opposite breakout)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals