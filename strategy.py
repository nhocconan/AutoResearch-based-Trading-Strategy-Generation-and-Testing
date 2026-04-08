#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_filter_v1
# Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and ATR stoploss.
# Long when price breaks above 20-period high + volume > 1.5x average volume.
# Short when price breaks below 20-period low + volume > 1.5x average volume.
# Uses daily trend filter to avoid counter-trend trades and reduce whipsaw.
# Designed for 20-50 trades/year on 4h to avoid fee drag. Works in bull/bear via daily trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period20_high = np.full(n, np.nan)
    period20_low = np.full(n, np.nan)
    for i in range(20, n):
        period20_high[i] = np.max(high[i-20:i+1])
        period20_low[i] = np.min(low[i-20:i+1])
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            avg_volume[i] = vol_sum / vol_count
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure Donchian is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low or trend changes
            if close[i] < period20_low[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high or trend changes
            if close[i] > period20_high[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirmed = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above 20-period high + volume confirmation + uptrend
            if (close[i] > period20_high[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period low + volume confirmation + downtrend
            elif (close[i] < period20_low[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals