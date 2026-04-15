# [EXPERIMENT #45917] 4h Donchian Breakout + Volume Confirmation + 1d Trend Filter
# Hypothesis: In both bull and bear markets, price often breaks out of consolidation ranges with volume.
# We use 4h Donchian(20) for structure, volume spike for conviction, and 1d EMA50 for trend filter.
# Long when price breaks above Donchian high + volume > 1.5x avg + price > 1d EMA50.
# Short when price breaks below Donchian low + volume > 1.5x avg + price < 1d EMA50.
# Exit when price returns to Donchian midpoint or trend fails.
# Discrete sizing (0.25) to limit trade frequency and fee drag.
# Target: 20-50 trades/year per symbol.

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
    
    # 4h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current > 1.5x 20-period median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high + volume + 1d uptrend
        if (close[i] > highest_high[i] and 
            volume[i] > vol_threshold[i] and 
            close[i] > ema_1d_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + volume + 1d downtrend
        elif (close[i] < lowest_low[i] and 
              volume[i] > vol_threshold[i] and 
              close[i] < ema_1d_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to Donchian midpoint or trend fails
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