#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation.
# Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 2x 20-period average.
# Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 2x 20-period average.
# Exit when price returns to Donchian middle band (20-period SMA) OR trend reversal.
# Uses 4h timeframe for optimal trade frequency (target: 75-200 total trades over 4 years).
# Donchian provides clear structure, 1d EMA50 filters for higher-timeframe trend, volume spike confirms conviction.
# Works in bull via trend-following breakouts, bear via faded rallies and mean-reversion to the middle band.

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
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
    
    # Calculate Donchian channels on 4h data
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    middle = (upper + lower) / 2  # 20-period SMA of midpoint
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_window, n):  # Start after sufficient data for Donchian
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper band AND price > 1d EMA50 AND volume confirmation
            if close[i] > upper[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower band AND price < 1d EMA50 AND volume confirmation
            elif close[i] < lower[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to middle band OR trend reversal (price < 1d EMA50)
            if close[i] <= middle[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to middle band OR trend reversal (price > 1d EMA50)
            if close[i] >= middle[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals