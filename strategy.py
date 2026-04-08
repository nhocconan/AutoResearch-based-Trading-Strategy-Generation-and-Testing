#!/usr/bin/env python3
# 6h_1w_1d_volume_price_action_v1
# Hypothesis: 6h price action with volume confirmation and 1w trend filter.
# Long when price breaks above 6h high of last 20 bars with volume > 1.5x average and 1w uptrend.
# Short when price breaks below 6h low of last 20 bars with volume > 1.5x average and 1w downtrend.
# Uses volume surge to confirm breakout strength and weekly trend to avoid counter-trend trades.
# Designed for 15-35 trades/year on 6h to minimize fee drag while capturing strong momentum moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_volume_price_action_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6s Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i+1])
        lowest_low[i] = np.min(low[i-20:i+1])
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20.0
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA25 for trend filter
    ema25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema25_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current volume > 1.5x 20-period average
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # 1w trend filter
        uptrend_1w = close[i] > ema25_1w_aligned[i]
        downtrend_1w = close[i] < ema25_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low or volume drops
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high or volume drops
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period high with volume surge and 1w uptrend
            if (close[i] > highest_high[i] and 
                volume_surge and 
                uptrend_1w):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period low with volume surge and 1w downtrend
            elif (close[i] < lowest_low[i] and 
                  volume_surge and 
                  downtrend_1w):
                position = -1
                signals[i] = -0.25
    
    return signals