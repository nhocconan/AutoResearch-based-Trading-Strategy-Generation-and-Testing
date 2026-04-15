#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ATR filter + volume confirmation
# Uses Donchian channels (20-period high/low) on 12h timeframe to detect breakouts.
# Long when price breaks above 20-period high, short when breaks below 20-period low.
# Only trade if 1d ATR > 1.5x 20-period median ATR (volatility filter).
# Volume must be > 1.5x 20-period median volume for confirmation.
# Designed to work in trending markets (both bull and bear) with volatility filter to avoid chop.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.
# Target: 15-30 trades/year on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR (14-period) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d ATR median for volatility threshold
    atr_median = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=1).median()
    atr_threshold = 1.5 * atr_median
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_threshold[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high, volatility filter, volume spike
        if (close[i] > donchian_high[i-1] and  # breakout above previous high
            atr_1d_aligned[i] > atr_threshold[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low, volatility filter, volume spike
        elif (close[i] < donchian_low[i-1] and  # breakout below previous low
              atr_1d_aligned[i] > atr_threshold[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel or volatility drops
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_ATR_Volume"
timeframe = "12h"
leverage = 1.0