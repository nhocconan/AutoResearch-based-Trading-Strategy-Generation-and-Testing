#!/usr/bin/env python3
# 4h_volume_price_action_v3
# Hypothesis: 4H price breaks Donchian(20) channel with volume confirmation and 1D trend filter.
# Works in bull (breakouts catch momentum) and bear (mean reversion at channel extremes).
# Uses tight entry conditions to limit trades and reduce fee drag. Target: 25-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_price_action_v3"
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
    
    # Donchian channel (20-period high/low)
    donch_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donch_period-1, n):
        upper[i] = np.max(high[i-donch_period+1:i+1])
        lower[i] = np.min(low[i-donch_period+1:i+1])
    
    # Volume filter: 2.0x 20-period average (tighter to reduce trades)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 2.0 * vol_ma[i]
    
    # 1D trend filter: EMA(50) slope (requires 2-bar confirmation)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate slope: positive if current EMA > EMA 2 periods ago
    ema50_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(2, len(close_1d)):
        if not np.isnan(ema50_1d[i]) and not np.isnan(ema50_1d[i-2]):
            ema50_slope_1d[i] = ema50_1d[i] - ema50_1d[i-2]
    ema50_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_period, vol_ma_period, 2) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_slope_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below lower Donchian or volume drops
            if close[i] < lower[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian or volume drops
            if close[i] > upper[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long entry: Price breaks above upper Donchian, volume surge, 1D trend up
            if (close[i] > upper[i] and 
                vol_surge[i] and 
                ema50_slope_1d_aligned[i] > 0):
                position = 1
                signals[i] = 0.30
            # Short entry: Price breaks below lower Donchian, volume surge, 1D trend down
            elif (close[i] < lower[i] and 
                  vol_surge[i] and 
                  ema50_slope_1d_aligned[i] < 0):
                position = -1
                signals[i] = -0.30
    
    return signals