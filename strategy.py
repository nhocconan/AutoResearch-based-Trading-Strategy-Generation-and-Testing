#!/usr/bin/env python3
# 12h_donchian_breakout_volume_v1
# Hypothesis: Donchian(20) breakout on 12h with volume confirmation and 1d trend filter.
# Long when: price breaks above 12h Donchian upper, 1d SMA50 rising, volume > 1.5x average.
# Short when: price breaks below 12h Donchian lower, 1d SMA50 falling, volume > 1.5x average.
# Exit when price returns to 12h Donchian midpoint or volume drops below average.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    donch_len = 20
    # Calculate upper and lower bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = low_series.rolling(window=donch_len, min_periods=donch_len).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for trend filter (SMA50 slope)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    # Calculate slope: positive if current SMA > SMA 2 periods ago
    sma50_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(2, len(close_1d)):
        if not np.isnan(sma50_1d[i]) and not np.isnan(sma50_1d[i-2]):
            sma50_slope_1d[i] = sma50_1d[i] - sma50_1d[i-2]
    sma50_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_len, vol_ma_period, 2) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(sma50_slope_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Donchian midpoint or volume drops below average
            if close[i] < donch_mid[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian midpoint or volume drops below average
            if close[i] > donch_mid[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above Donchian upper, 1d SMA50 slope positive, volume surge
            if (close[i] > donch_high[i] and 
                sma50_slope_1d_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian lower, 1d SMA50 slope negative, volume surge
            elif (close[i] < donch_low[i] and 
                  sma50_slope_1d_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals