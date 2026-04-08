#!/usr/bin/env python3
# 12h_donchian_volume_trend_v1
# Hypothesis: 12h Donchian channel breakouts with volume confirmation and daily trend filter.
# In bull markets, buy breakouts above upper channel with rising volume and daily uptrend.
# In bear markets, sell breakdowns below lower channel with rising volume and daily downtrend.
# Uses tight entry conditions to limit trades and reduce fee drag.

name = "12h_donchian_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    dc_period = 20
    upper_channel = np.full_like(high, np.nan)
    lower_channel = np.full_like(low, np.nan)
    
    for i in range(dc_period - 1, n):
        upper_channel[i] = np.max(high[i-dc_period+1:i+1])
        lower_channel[i] = np.min(low[i-dc_period+1:i+1])
    
    # Volume filter: 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_ma[:19] = vol_ma[19]  # Fill beginning
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_daily = np.full_like(close_daily, np.nan)
    for i in range(ema_period - 1, len(close_daily)):
        ema_daily[i] = np.mean(close_daily[i-ema_period+1:i+1])
    
    # Align daily EMA to 12h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(dc_period, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below lower channel or trend reverses
            if close[i] < lower_channel[i] or not uptrend_htf:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above upper channel or trend reverses
            if close[i] > upper_channel[i] or not downtrend_htf:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with volume and daily uptrend
            if (close[i] > upper_channel[i] and 
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with volume and daily downtrend
            elif (close[i] < lower_channel[i] and 
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals