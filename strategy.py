#!/usr/bin/env python3
"""
4h_donchian_20_1d_volume_v2
Hypothesis: On 4-hour timeframe, enter long when price breaks above 20-period Donchian high with volume confirmation and 1d trend filter.
Enter short when price breaks below 20-period Donchian low with volume confirmation and 1d trend filter.
Exit when price crosses the 20-period Donchian midpoint.
Uses 1d EMA50 as trend filter to avoid counter-trend trades. Volume > 1.5x 20-period average confirms institutional interest.
Designed for low trade frequency (20-40/year) to minimize fee drag while capturing sustained trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50 = pd.Series(d_close).ewm(span=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit when price crosses below Donchian midpoint
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price crosses above Donchian midpoint
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high + uptrend + volume
            long_entry = (close[i] > donch_high[i]) and uptrend and vol_confirmed
            
            # Short entry: price breaks below Donchian low + downtrend + volume
            short_entry = (close[i] < donch_low[i]) and downtrend and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals