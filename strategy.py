#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h and 1d filters. Uses 4h EMA20 for trend direction and 1d Donchian breakouts for entry timing.
In uptrend (price > 4h EMA20): long when price breaks above 1d Donchian upper band (20-period).
In downtrend (price < 4h EMA20): short when price breaks below 1d Donchian lower band (20-period).
Volume must be above 20-period average to confirm breakout.
Session filter: only trade 08-20 UTC to avoid low-liquidity hours.
Fixed position size: 0.20 (20% of capital).
Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema20_donchian_breakout_1d_vol_session_v1"
timeframe = "1h"
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
    
    # === 4H EMA20 TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    ema_4h_close = df_4h['close'].values
    ema_4h = pd.Series(ema_4h_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # already shifted
    
    # === 1D DONCHIAN CHANNELS (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    donchian_high = df_1d['high'].rolling(window=20, min_periods=20).max().values
    donchian_low = df_1d['low'].rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === VOLUME CONFIRMATION (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === SESSION FILTER: 08-20 UTC ===
    # Pre-compute hour from DatetimeIndex (already datetime64[ns])
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for 20-period indicators
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h EMA20
        uptrend = close[i] > ema_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 4h EMA20 OR donchian lower band
            if close[i] < ema_4h_aligned[i] or close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above 4h EMA20 OR donchian upper band
            if close[i] > ema_4h_aligned[i] or close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 4h trend
            if uptrend:
                # In uptrend: long on break above 1d donchian high
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.20
            else:
                # In downtrend: short on break below 1d donchian low
                if close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals