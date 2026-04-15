#!/usr/bin/env python3
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
    
    # 12h price channel (Donchian 20-period)
    high_12h = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_12h = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Weekly trend filter: 20-period EMA
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily volume spike: current > 2.0x median of last 50 periods
    df_1d = get_htf_data(prices, '1d')
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=50, min_periods=50).median()
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_median_1d_aligned[i])):
            continue
        
        # Long: price breaks above 12h high + above weekly EMA + volume spike
        if (close[i] > high_12h[i] and 
            close[i] > ema_20_1w_aligned[i] and 
            volume[i] > 2.0 * vol_median_1d_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below 12h low + below weekly EMA + volume spike
        elif (close[i] < low_12h[i] and 
              close[i] < ema_20_1w_aligned[i] and 
              volume[i] > 2.0 * vol_median_1d_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to midline of 12h channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (high_12h[i] + low_12h[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (high_12h[i] + low_12h[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_WeeklyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0