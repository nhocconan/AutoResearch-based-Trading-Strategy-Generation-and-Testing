#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + Volume Confirmation + 1d Trend Filter
# Long when price breaks above Donchian(20) high + volume spike + 1d EMA50 uptrend
# Short when price breaks below Donchian(20) low + volume spike + 1d EMA50 downtrend
# Exit when price reverses to Donchian midpoint or trend fails
# Works in bull (breakouts with volume) and bear (breakdowns with volume)
# Discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: breakout above Donchian high + volume + 1d uptrend
        if (close[i] > high_max[i] and 
            volume[i] > vol_threshold[i] and 
            close[i] > ema_1d_aligned[i]):
            signals[i] = 0.25
        
        # Short: breakdown below Donchian low + volume + 1d downtrend
        elif (close[i] < low_min[i] and 
              volume[i] > vol_threshold[i] and 
              close[i] < ema_1d_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price reverses to midpoint or trend fails
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= donchian_mid[i] or close[i] <= ema_1d_aligned[i])) or
               (signals[i-1] == -0.25 and (close[i] >= donchian_mid[i] or close[i] >= ema_1d_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0