#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_filter_v1
# Hypothesis: Daily Donchian(20) breakout with volume confirmation and weekly trend filter.
# Long when price breaks above 20-day high with volume > 1.5x average and weekly uptrend.
# Short when price breaks below 20-day low with volume > 1.5x average and weekly downtrend.
# Designed for 10-30 trades/year on 1d to avoid fee drag. Works in bull/bear via weekly trend alignment.

import numpy as np
import pandas as pd
from mft_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_filter_v1"
timeframe = "1d"
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
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_avg[i] = vol_sum / 20.0
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA25 for trend filter
    ema25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure Donchian is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema25_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x average
        vol_condition = volume[i] > 1.5 * vol_avg[i]
        
        # Weekly trend filter
        uptrend_1w = close[i] > ema25_1w_aligned[i]
        downtrend_1w = close[i] < ema25_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low
            if close[i] < period20_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high
            if close[i] > period20_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-day high with volume confirmation and weekly uptrend
            if (close[i] > period20_high[i] and 
                vol_condition and 
                uptrend_1w):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-day low with volume confirmation and weekly downtrend
            elif (close[i] < period20_low[i] and 
                  vol_condition and 
                  downtrend_1w):
                position = -1
                signals[i] = -0.25
    
    return signals