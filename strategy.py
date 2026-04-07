#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_trend_volume_spike"
timeframe = "1d"
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
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50 EMA on weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily EMA for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # Volume spike filter: volume > 2x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if EMA or volume data not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 20 EMA
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 20 EMA
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Require volume spike for entry
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
            
            # Long entry: price above weekly EMA50 AND crosses above daily EMA20
            if close[i] > ema_50_1w_aligned[i] and close[i] > ema_20[i] and close[i-1] <= ema_20[i-1]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below weekly EMA50 AND crosses below daily EMA20
            elif close[i] < ema_50_1w_aligned[i] and close[i] < ema_20[i] and close[i-1] >= ema_20[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals