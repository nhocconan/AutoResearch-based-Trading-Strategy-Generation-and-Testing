#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Volume Spike and Weekly EMA Trend Filter
Target: 50-150 trades over 4 years (12-37/year) for low fee drag.
Uses Donchian(20) breakout in direction of weekly EMA34 trend, confirmed by
1d volume spike (>2x average). Works in bull/bear by following higher timeframe
trend. Position size: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike detection
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    # We need to calculate on the actual price series since we're using 12h timeframe
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # Volume spike detection on 1d: volume > 2x 4-period average
    vol_ma_1d = pd.Series(vol_1d).rolling(window=4, min_periods=4).mean().values
    vol_spike_1d = align_htf_to_ltf(prices, df_1d, vol_1d > (2.0 * vol_ma_1d))
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = max(50, lookback)  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_spike_1d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = vol_spike_1d[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and above weekly EMA
            if (price > upper and 
                vol_spike and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower with volume spike and below weekly EMA
            elif (price < lower and 
                  vol_spike and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: hold until reverse signal
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower
            if price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until reverse signal
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper
            if price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_Volume_Spike_1wEMA34"
timeframe = "12h"
leverage = 1.0