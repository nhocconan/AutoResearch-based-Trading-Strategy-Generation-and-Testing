#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d daily pivot direction + volume confirmation
# Uses daily pivot levels to filter breakouts in the direction of daily bias.
# Long when price breaks above Donchian(20) high AND daily pivot > open (bullish bias).
# Short when price breaks below Donchian(20) low AND daily pivot < open (bearish bias).
# Volume confirmation requires > 1.5x 20-bar median volume.
# Designed to work in trending markets by following daily bias.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day pivot point (PP) and bias direction
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Pivot point = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Daily bias: bullish if PP > open, bearish if PP < open
    bias_bullish = pp_1d > open_1d
    bias_bearish = pp_1d < open_1d
    
    # Align bias to 6h timeframe
    bias_bullish_aligned = align_htf_to_ltf(prices, df_1d, bias_bullish.astype(float))
    bias_bearish_aligned = align_htf_to_ltf(prices, df_1d, bias_bearish.astype(float))
    
    # Donchian(20) channels on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(bias_bullish_aligned[i]) or np.isnan(bias_bearish_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: break above Donchian high AND daily bullish bias AND volume spike
        if (close[i] > donchian_high[i] and 
            bias_bullish_aligned[i] > 0.5 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: break below Donchian low AND daily bearish bias AND volume spike
        elif (close[i] < donchian_low[i] and 
              bias_bearish_aligned[i] > 0.5 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel or bias flips
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= donchian_high[i] or bias_bullish_aligned[i] <= 0.5)) or
               (signals[i-1] == -0.25 and (close[i] >= donchian_low[i] or bias_bearish_aligned[i] <= 0.5)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Donchian_Pivot_Bias_Volume"
timeframe = "6h"
leverage = 1.0