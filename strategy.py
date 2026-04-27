#!/usr/bin/env python3
"""
#100929 - 4h_Donchian20_1dTrend_Volume_Slope
Hypothesis: Donchian(20) breakout with 1d EMA trend filter and volume confirmation on 4h timeframe.
Adds price slope filter to reduce false breakouts. Targets 20-50 trades/year (80-200 total) to minimize fee drift.
Works in bull (breakouts with trend) and bear (mean reversion via stop/reversal).
"""

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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate price slope (5-period linear regression slope)
    def rolling_slope(series, window):
        slopes = np.full_like(series, np.nan, dtype=np.float64)
        for i in range(window - 1, len(series)):
            y = series[i - window + 1:i + 1]
            x = np.arange(window)
            if np.all(np.isnan(y)):
                slopes[i] = np.nan
            else:
                # Use polyfit for slope (degree 1)
                coeffs = np.polyfit(x, y, 1)
                slopes[i] = coeffs[0]
        return slopes
    
    price_slope = rolling_slope(close, 5)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(price_slope[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, above 1d EMA34, positive slope, volume spike
        if (close[i] > high_20[i] and 
            close[i] > ema34_1d_aligned[i] and 
            price_slope[i] > 0 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian low, below 1d EMA34, negative slope, volume spike
        elif (close[i] < low_20[i] and 
              close[i] < ema34_1d_aligned[i] and 
              price_slope[i] < 0 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite Donchian level or opposite EMA
        elif position == 1 and (close[i] < low_20[i] or close[i] < ema34_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > high_20[i] or close[i] > ema34_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dTrend_Volume_Slope"
timeframe = "4h"
leverage = 1.0