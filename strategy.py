#!/usr/bin/env python3
# 1h_donchian_breakout_4h_trend_volume_v1
# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation.
# Uses 4h Donchian channels for trend direction and 1h breakouts for entry timing.
# Volume filter ensures participation. Designed for low trade frequency (15-37/year) to avoid fee drag.
# Works in bull markets via breakouts and bear markets via short breakdowns.

name = "1h_donchian_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian for trend direction (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_period = 20
    dc_upper_4h = np.full_like(high_4h, np.nan)
    dc_lower_4h = np.full_like(low_4h, np.nan)
    
    for i in range(donchian_period - 1, len(high_4h)):
        dc_upper_4h[i] = np.max(high_4h[i - donchian_period + 1:i + 1])
        dc_lower_4h[i] = np.min(low_4h[i - donchian_period + 1:i + 1])
    
    # Forward fill to handle NaNs
    dc_upper_4h = pd.Series(dc_upper_4h).ffill().bfill().values
    dc_lower_4h = pd.Series(dc_lower_4h).ffill().bfill().values
    
    # Align to 1h
    dc_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_upper_4h)
    dc_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_lower_4h)
    
    # 1h Donchian for entry timing (20-period)
    dc_upper_1h = np.full_like(high, np.nan)
    dc_lower_1h = np.full_like(low, np.nan)
    
    for i in range(donchian_period - 1, len(high)):
        dc_upper_1h[i] = np.max(high[i - donchian_period + 1:i + 1])
        dc_lower_1h[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Forward fill to handle NaNs
    dc_upper_1h = pd.Series(dc_upper_1h).ffill().bfill().values
    dc_lower_1h = pd.Series(dc_lower_1h).ffill().bfill().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.zeros_like(volume)
    vol_ma[vol_period - 1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period - 1:].values
    vol_ma[:vol_period - 1] = vol_ma[vol_period - 1]
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Start from sufficient lookback
    start_idx = max(donchian_period, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if session filter fails
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(dc_upper_1h[i]) or np.isnan(dc_lower_1h[i]) or 
            np.isnan(dc_upper_4h_aligned[i]) or np.isnan(dc_lower_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below 4h lower Donchian or volume fails
            if close[i] < dc_lower_4h_aligned[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit if price breaks above 4h upper Donchian or volume fails
            if close[i] > dc_upper_4h_aligned[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: break above 1h upper Donchian with 4h uptrend and volume
            if (close[i] > dc_upper_1h[i] and 
                close[i] > dc_upper_4h_aligned[i] and  # 4h uptrend filter
                volume_filter):
                position = 1
                signals[i] = 0.20
            # Short entry: break below 1h lower Donchian with 4h downtrend and volume
            elif (close[i] < dc_lower_1h[i] and 
                  close[i] < dc_lower_4h_aligned[i] and  # 4h downtrend filter
                  volume_filter):
                position = -1
                signals[i] = -0.20
    
    return signals