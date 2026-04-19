#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
# In bull markets, breakouts capture momentum; in bear markets, 1d ATR filter avoids
# false breakouts during low volatility, reducing whipsaw. Volume confirms institutional interest.
# Target: 20-40 trades/year per symbol, low turnover, high edge.
name = "4h_Donchian20_1dATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Use previous day's ATR to avoid look-ahead
    atr_14_prev = np.concatenate([[np.nan], atr_14[:-1]])
    atr_14_prev_aligned = align_htf_to_ltf(prices, df_1d, atr_14_prev)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback, 20)
    
    for i in range(start_idx, n):
        if np.isnan(atr_14_prev_aligned[i]) or np.isnan(highest_high[i]) or \
           np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr_14_prev_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and ATR filter
            if price > highest_high[i] and volume_ok and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and ATR filter
            elif price < lowest_low[i] and volume_ok and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below Donchian low or volatility drops
            if price < lowest_low[i] or atr_val < 0.5 * atr_14_prev_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above Donchian high or volatility drops
            if price > highest_high[i] or atr_val < 0.5 * atr_14_prev_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals