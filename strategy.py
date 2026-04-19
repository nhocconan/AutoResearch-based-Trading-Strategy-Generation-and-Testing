#!/usr/bin/env python3
"""
4h_Donchian20_1dVolumeSpike_1wEMA34
Hypothesis: 4h Donchian(20) breakout with weekly EMA34 trend filter and daily volume spike.
In bull markets: price above weekly EMA34, break above upper Donchian + volume spike = long.
In bear markets: price below weekly EMA34, break below lower Donchian + volume spike = short.
Volume spike confirms institutional participation, reducing false breakouts.
Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
Works in bull/bear via weekly EMA34 trend filter and volume confirmation.
"""

name = "4h_Donchian20_1dVolumeSpike_1wEMA34"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume spike: volume > 2.0 * 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    volume_spike = volume > (vol_20d_aligned * 2.0)
    
    # 4h Donchian(20) channels
    lookback = 20
    upper = np.full_like(close, np.nan)
    lower = np.full_like(close, np.nan)
    for i in range(lookback-1, len(high)):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback-1, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price above weekly EMA34, breaks above upper Donchian, volume spike
            if (price_above_ema and 
                close[i] > upper[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34, breaks below lower Donchian, volume spike
            elif (price_below_ema and 
                  close[i] < lower[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian or crosses below weekly EMA34
            if (close[i] < lower[i]) or (not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian or crosses above weekly EMA34
            if (close[i] > upper[i]) or (not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals