#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
# Works in bull/bear: breakouts capture momentum, ATR filter avoids chop, volume confirms strength
# Target: 20-40 trades/year, low frequency to minimize fee drag
name = "4h_Donchian20_ATR1d_Volume_Filter_v1"
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
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # ATR filter: avoid low volatility (choppy) markets
        atr_filter = atr > 0.01 * price  # ATR > 1% of price
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above upper Donchian with volume and ATR filter
            if price > highest_high[i] and volume_ok and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and ATR filter
            elif price < lowest_low[i] and volume_ok and atr_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to midpoint of Donchian channel
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals