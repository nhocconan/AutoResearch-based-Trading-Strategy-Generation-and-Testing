#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 6h Donchian upper(20) AND price > 1d EMA50 AND volume > 1.5x average
# Short when price breaks below 6h Donchian lower(20) AND price < 1d EMA50 AND volume > 1.5x average
# Exit when price crosses the 6h Donchian midpoint (mean reversion) OR trend reversal
# Uses 6h timeframe for lower frequency, Donchian for structure, 1d EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies.

name = "6h_Donchian20_1dTrend_Volume_v1"
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
    
    # Get 6h data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Donchian channels (20-period) on 6h
    high_max_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (1.5 * vol_ma_6h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above upper Donchian AND price > 1d EMA50 AND volume confirmation
            if close[i] > high_max_20[i] and close[i] > ema50_1d_aligned[i] and volume_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower Donchian AND price < 1d EMA50 AND volume confirmation
            elif close[i] < low_min_20[i] and close[i] < ema50_1d_aligned[i] and volume_filter_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian midpoint OR trend reversal (price < 1d EMA50)
            if close[i] < donchian_mid[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian midpoint OR trend reversal (price > 1d EMA50)
            if close[i] > donchian_mid[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals