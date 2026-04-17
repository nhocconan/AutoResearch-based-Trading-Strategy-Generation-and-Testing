#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above Donchian upper band AND volume > 1.5x 20-period average AND 12h EMA34 > EMA55.
Short when price breaks below Donchian lower band AND volume > 1.5x 20-period average AND 12h EMA34 < EMA55.
Exit when price crosses Donchian middle band (20-period midpoint) or volume drops below average.
Uses proven Donchian breakout structure with volume and trend filters to reduce false signals.
Designed for 4h timeframe targeting 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on primary timeframe (4h)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_band = (highest_high + lowest_low) / 2.0
    
    # Calculate volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema55_12h = pd.Series(close_12h).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align 12h indicators to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema55_12h_aligned = align_htf_to_ltf(prices, df_12h, ema55_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(lookback, 20, 34, 55)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(middle_band[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema34_12h_aligned[i]) or np.isnan(ema55_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        ema34 = ema34_12h_aligned[i]
        ema55 = ema55_12h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper band AND volume > 1.5x avg AND 12h EMA34 > EMA55 (uptrend)
            if price > highest_high[i] and vol > 1.5 * vol_ma and ema34 > ema55:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band AND volume > 1.5x avg AND 12h EMA34 < EMA55 (downtrend)
            elif price < lowest_low[i] and vol > 1.5 * vol_ma and ema34 < ema55:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below middle band OR volume drops below average
            if price < middle_band[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above middle band OR volume drops below average
            if price > middle_band[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMATrend"
timeframe = "4h"
leverage = 1.0