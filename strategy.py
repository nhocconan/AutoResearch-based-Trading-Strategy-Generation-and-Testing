#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and 1-week ATR filter.
# Strategy buys when price breaks above 20-period Donchian high with above-average volume,
# sells when price breaks below 20-period Donchian low with above-average volume.
# Uses 1-week ATR to filter out low-volatility choppy markets (ATR < 50-day SMA of ATR).
# Works in both bull and bear markets by trading breakouts in the direction of volatility expansion.
# Target: 20-50 trades per year to minimize fee drag while capturing significant moves.
name = "4h_Donchian20_1dVolume_1wATRFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period average volume on 1d
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 1w data for ATR filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ATR on 1w data
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w.shift(1))
    tr3 = np.abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean()
    
    # 50-period SMA of ATR for filtering
    atr_ma_50_1w = atr_1w.rolling(window=50, min_periods=50).mean()
    
    # Align ATR and its MA to lower timeframe (4h)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w.values)
    atr_ma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_50_1w.values)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_ma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1d average volume
        volume_confirm = volume[i] > vol_ma_20_1d_aligned[i]
        
        # Volatility filter: current ATR > 50-day SMA of ATR (avoid choppy markets)
        vol_filter = atr_1w_aligned[i] > atr_ma_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + vol filter
            if close[i] > highest_high[i] and volume_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + vol filter
            elif close[i] < lowest_low[i] and volume_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals