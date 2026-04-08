#!/usr/bin/env python3
# 6h_volume_breakout_1d_trend_v1
# Hypothesis: Breakout strategy combining 6h Donchian breakout with 1d trend filter (EMA50) and volume confirmation.
# Enter long when price breaks above 20-period Donchian high, price > 1d EMA50, and volume > 1.5x average volume.
# Enter short when price breaks below 20-period Donchian low, price < 1d EMA50, and volume > 1.5x average volume.
# Exit when price returns to the Donchian midpoint or trend filter fails.
# Designed for 20-40 trades/year on 6h to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_breakout_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint or trend filter fails
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < midpoint or close[i] <= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint or trend filter fails
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > midpoint or close[i] >= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high with volume and trend filter
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low with volume and trend filter
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals