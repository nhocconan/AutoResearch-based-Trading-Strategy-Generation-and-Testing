#!/usr/bin/env python3
"""
1d_momentum_volume_1w_trend_v1
Hypothesis: On 1-day timeframe, combine weekly trend filter (EMA20) with momentum breakout and volume confirmation. 
Long when: price breaks above 20-day high with weekly EMA uptrend and volume spike. 
Short when: price breaks below 20-day low with weekly EMA downtrend and volume spike.
Exit on opposite breakout or trend change.
Designed for 30-100 total trades over 4 years (~7-25/year) to minimize fee dust while capturing momentum in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_momentum_volume_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA(20) to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 20-day high/low for breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_20 = high_series.rolling(window=20, min_periods=20).max().values
    low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low or weekly trend turns down
            if low[i] <= low_20[i] or ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high or weekly trend turns up
            if high[i] >= high_20[i] or ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok:
                # Momentum breakout: new 20-day high/low with weekly trend alignment
                # Long: break above 20-day high with weekly uptrend
                if high[i] >= high_20[i] and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                    position = 1
                    signals[i] = 0.30
                # Short: break below 20-day low with weekly downtrend
                elif low[i] <= low_20[i] and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                    position = -1
                    signals[i] = -0.30
    
    return signals