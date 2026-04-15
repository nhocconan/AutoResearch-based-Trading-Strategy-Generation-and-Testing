#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h EMA50 trend + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 with volume > 1.5x median
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 with volume > 1.5x median
# Exit when price crosses back through Donchian midpoint
# Uses tight entry conditions to limit trades and avoid fee drag, targeting 20-50 trades/year

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50.values)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    mid_20 = (high_20 + low_20) / 2
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: break above upper Donchian + above 12h EMA50 + volume spike
        if (close[i] > high_20[i] and 
            close[i] > ema_50_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: break below lower Donchian + below 12h EMA50 + volume spike
        elif (close[i] < low_20[i] and 
              close[i] < ema_50_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back through Donchian midpoint
        elif i > 0:
            if signals[i-1] == 0.25 and close[i] < mid_20[i]:
                signals[i] = 0.0
            elif signals[i-1] == -0.25 and close[i] > mid_20[i]:
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0