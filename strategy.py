#!/usr/bin/env python3
"""
1h_4h_1d_trend_follow_volume_v1
Hypothesis: Use 4h and 1d trends to determine direction, 1h for entry timing with volume confirmation.
Trades only in 08-20 UTC to avoid low-volume periods. Targets 15-37 trades/year.
Uses 4h EMA(21) and 1d EMA(50) for trend, with volume > 1.5x 20-period average for entry.
Long when price > both EMAs, short when price < both EMAs, with volume filter.
Position size fixed at 0.20 to manage risk. Designed for both bull and bear markets via trend following.
"""

name = "1h_4h_1d_trend_follow_volume_v1"
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
    
    # Get 4h data for EMA(21) trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for EMA(50) trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMAs
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Start from sufficient lookback
    start_idx = max(21, 50, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below either EMA (trend failure)
            if close[i] < ema_4h_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit if price breaks above either EMA (trend failure)
            if close[i] > ema_4h_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: price above both EMAs with volume
            if close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.20
            # Short entry: price below both EMAs with volume
            elif close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.20
    
    return signals