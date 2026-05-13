#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume spike confirmation.
# Long when price breaks above 6h Donchian upper(20) AND 12h EMA50 is rising AND volume > 2x 20-period average.
# Short when price breaks below 6h Donchian lower(20) AND 12h EMA50 is falling AND volume > 2x 20-period average.
# Exit when price touches the 6h Donchian midpoint (mean reversion within the channel) OR volume dries up.
# Uses 6h timeframe for lower frequency, Donchian for structure, 12h EMA for trend, volume for conviction.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies.

name = "6h_Donchian20_12hTrend_Volume_v1"
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
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Donchian channels on 6h: upper(20), lower(20), midpoint
    high_series_6h = pd.Series(high_6h)
    low_series_6h = pd.Series(low_6h)
    donchian_upper = high_series_6h.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series_6h.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume filter: current 6h volume > 2x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (2.0 * vol_ma_6h)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter and slope
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    # Calculate slope of EMA50: rising if current > previous, falling if current < previous
    ema50_slope = np.diff(ema50_12h_aligned, prepend=ema50_12h_aligned[0])
    ema50_rising = ema50_slope > 0
    ema50_falling = ema50_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above Donchian upper AND 12h EMA50 rising AND volume spike
            if close[i] > donchian_upper[i] and ema50_rising[i] and volume_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian lower AND 12h EMA50 falling AND volume spike
            elif close[i] < donchian_lower[i] and ema50_falling[i] and volume_filter_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches Donchian midpoint OR volume dries up
            if close[i] >= donchian_mid[i] or not volume_filter_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches Donchian midpoint OR volume dries up
            if close[i] <= donchian_mid[i] or not volume_filter_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals