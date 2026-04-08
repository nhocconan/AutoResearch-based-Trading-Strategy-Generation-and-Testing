#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# In bull markets: buy breakouts above upper band in uptrend.
# In bear markets: sell breakdowns below lower band in downtrend.
# Volume filter ensures breakouts have institutional participation.
# Designed for low trade frequency (<50/year) to minimize fee drag in ranging/chaff markets.

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    highest_high[lookback-1:] = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()[lookback-1:].values
    lowest_low[lookback-1:] = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()[lookback-1:].values
    
    # 1d EMA trend filter (50-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(lookback, 50) + vol_period
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend fails
            if close[i] < lowest_low[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend fails
            if close[i] > highest_high[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price closes above Donchian upper with uptrend + volume
            if (close[i] > highest_high[i] and close[i] > ema_daily_aligned[i] and
                volume_filter):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price closes below Donchian lower with downtrend + volume
            elif (close[i] < lowest_low[i] and close[i] < ema_daily_aligned[i] and
                  volume_filter):
                position = -1
                signals[i] = -0.25
    
    return signals