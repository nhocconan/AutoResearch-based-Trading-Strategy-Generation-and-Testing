#!/usr/bin/env python3
# 4h_1w_1d_volume_breakout_v2
# Hypothesis: Breakout strategy using weekly trend filter (EMA50) and daily volume confirmation.
# Enter long when price breaks above 4h 20-period Donchian high, price > weekly EMA50, and daily volume > 1.5x average daily volume.
# Enter short when price breaks below 4h 20-period Donchian low, price < weekly EMA50, and daily volume > 1.5x average daily volume.
# Exit when price returns to the Donchian midpoint or weekly trend filter fails.
# Designed for 15-35 trades/year on 4h to avoid fee drag. Weekly trend filter ensures robustness in bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_1d_volume_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume average (20-period) for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(60, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average daily volume
        vol_confirmed = volume_1d[i] > 1.5 * vol_avg_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint or trend filter fails
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < midpoint or close[i] <= ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint or trend filter fails
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > midpoint or close[i] >= ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high with volume and trend filter
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1w_aligned[i] and 
                vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low with volume and trend filter
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals