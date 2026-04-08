#!/usr/bin/env python3
# 4h_12h_1d_volume_breakout_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h trend filter.
# Long when price breaks above Donchian high + volume > 1.5x avg + 12h EMA50 up.
# Short when price breaks below Donchian low + volume > 1.5x avg + 12h EMA50 down.
# Exit when price returns to Donchian midpoint or trend fails.
# Designed for 20-50 trades/year on 4h to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) for confirmation
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint or trend fails
            if close[i] < donch_mid[i] or close[i] <= ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint or trend fails
            if close[i] > donch_mid[i] or close[i] >= ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high with volume and trend filter
            if (close[i] > donch_high[i] and 
                vol_confirmed and 
                close[i] > ema50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low with volume and trend filter
            elif (close[i] < donch_low[i] and 
                  vol_confirmed and 
                  close[i] < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals