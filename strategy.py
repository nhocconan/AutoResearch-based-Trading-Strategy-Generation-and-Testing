#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily EMA trend and volume confirmation
# Uses daily EMA(50) for trend direction, 12h Donchian(20) breakout for entries,
# and volume > 1.5x 20-period average for confirmation.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear via mean reversion at channel extremes.

name = "12h_donchian20_daily_ema_volume_v2"
timeframe = "12h"
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
    
    # Daily EMA(50) for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Long: price breaks above Donchian upper band + volume + above daily EMA
        if (close[i] > highest_high[i-1] and vol_ok and 
            close[i] > ema_1d_aligned[i]):
            signals[i] = 0.25
        # Short: price breaks below Donchian lower band + volume + below daily EMA
        elif (close[i] < lowest_low[i-1] and vol_ok and 
              close[i] < ema_1d_aligned[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals