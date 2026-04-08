#!/usr/bin/env python3
# 12h_donchian20_volume_breakout_v1
# Hypothesis: Donchian(20) breakout on 12h timeframe with volume confirmation.
# Long when price breaks above 20-period high + volume > 1.5x average.
# Short when price breaks below 20-period low + volume > 1.5x average.
# Exit when price returns to the 20-period midpoint.
# Uses 1w trend filter to avoid counter-trend trades in strong trends.
# Target: 15-25 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # 1w trend filter (SMA50 slope)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        if not np.isnan(sma50_1w[i]) and not np.isnan(sma50_1w[i-1]):
            sma50_slope_1w[i] = sma50_1w[i] - sma50_1w[i-1]
    sma50_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(lookback, vol_ma_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma50_slope_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to midpoint or below
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to midpoint or above
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high + volume surge + 1w uptrend
            if (close[i] > donchian_high[i] and 
                vol_surge[i] and 
                sma50_slope_1w_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + volume surge + 1w downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_surge[i] and 
                  sma50_slope_1w_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals