#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_filter_v2
# Hypothesis: 1d Donchian breakout with volume confirmation and 1w trend filter.
# Long on breakout above 20-day high + volume > 1.5x avg + price above 10-week EMA.
# Short on breakdown below 20-day low + volume > 1.5x avg + price below 10-week EMA.
# Designed for 10-25 trades/year on 1d to avoid fee drag. Works in bull/bear via multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_filter_v2"
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
    
    # 1d Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i+1])
        donchian_low[i] = np.min(low[i-20:i+1])
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i+1])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA10 for trend filter
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema10_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high + volume confirmation + 1w uptrend
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                close[i] > ema10_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low + volume confirmation + 1w downtrend
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  close[i] < ema10_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals