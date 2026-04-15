#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout + 1w EMA Trend + Volume Spike
# Long when price breaks above Donchian(20) high, price > weekly EMA50, and volume > 2x median
# Short when price breaks below Donchian(20) low, price < weekly EMA50, and volume > 2x median
# Uses weekly trend filter to avoid counter-trend trades, volume confirmation to ensure momentum
# Designed for low-frequency, high-conviction trades to minimize fee drag in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high, above weekly EMA, volume spike
        if (close[i] > high_max[i] and 
            close[i] > ema_1w_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low, below weekly EMA, volume spike
        elif (close[i] < low_min[i] and 
              close[i] < ema_1w_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to opposite Donchian band or weekly EMA
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < low_min[i] or close[i] < ema_1w_aligned[i])) or
               (signals[i-1] == -0.25 and (close[i] > high_max[i] or close[i] > ema_1w_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_1wEMA_Volume"
timeframe = "12h"
leverage = 1.0