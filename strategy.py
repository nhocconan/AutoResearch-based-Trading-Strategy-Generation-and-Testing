#!/usr/bin/env python3
# 1h_4h_1d_trend_follow_volume_v1
# Hypothesis: Trend-following strategy using 4h/1d EMAs for direction and 1h for entry timing with volume confirmation.
# Designed for 1h timeframe to achieve 15-37 trades/year by using higher timeframe filters to reduce noise.
# Works in bull/bear markets via trend filters and volume confirmation to avoid false breakouts.

name = "1h_4h_1d_trend_follow_volume_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA trend filter (21-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA trend filter (55-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.5x 24-period average (1 day)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Pre-compute for efficiency
    
    # Start from sufficient lookback
    start_idx = max(55, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA or 1d EMA
            if close[i] < ema_4h_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA or 1d EMA
            if close[i] > ema_4h_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only trade during session with volume confirmation
            if in_session and volume_filter:
                # Entry long: price above both EMAs
                if close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Entry short: price below both EMAs
                elif close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals