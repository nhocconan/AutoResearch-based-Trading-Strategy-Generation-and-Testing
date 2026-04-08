#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: 4h Donchian channel breakout with 1d trend filter and volume confirmation.
# In both bull and bear markets, strong trends persist; breakouts from volatility contractions
# capture the start of new moves. Volume confirms institutional participation.
# Target: 20-40 trades/year with position size 0.25 to minimize fee drag.

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
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
    
    # Donchian Channel (20)
    dc_period = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    
    for i in range(dc_period - 1, n):
        dc_upper[i] = np.max(high[i-dc_period+1:i+1])
        dc_lower[i] = np.min(low[i-dc_period+1:i+1])
    
    # 1d EMA trend filter (50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(dc_period, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price re-enters Donchian Channel or trend fails
            if close[i] <= dc_upper[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price re-enters Donchian Channel or trend fails
            if close[i] >= dc_lower[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above upper Donchian with uptrend and volume
            if close[i] > dc_upper[i] and close[i] > ema_1d_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: break below lower Donchian with downtrend and volume
            elif close[i] < dc_lower[i] and close[i] < ema_1d_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals