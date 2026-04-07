#!/usr/bin/env python3
"""
6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
In bull market (weekly close > weekly EMA20): long on 20-bar high breakout.
In bear market (weekly close < weekly EMA20): short on 20-bar low breakout.
Volume must be above 20-period average to confirm breakout strength.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_direction_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)  # already shifted
    
    # === DONCHIAN CHANNEL (LTF) ===
    lookback = 20
    high_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        bull_trend = close[i] > weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < low_min[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > high_max[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on weekly trend
            if bull_trend:
                # In bull market: long on breakout above Donchian high
                if high[i] > high_max[i-1]:  # Use previous bar's high to avoid look-ahead
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on breakdown below Donchian low
                if low[i] < low_min[i-1]:  # Use previous bar's low to avoid look-ahead
                    position = -1
                    signals[i] = -0.25
    
    return signals