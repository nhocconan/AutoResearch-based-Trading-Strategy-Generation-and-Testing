#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_donchian_breakout_with_volume_and_atr
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period average + ATR(14) > 0.5 * ATR(50) (volatility filter).
# Short when price breaks below 20-period Donchian low + same volume and volatility filters.
# Exit when price crosses the 20-period Donchian midpoint (mean reversion).
# Uses 1d ATR for volatility regime filter to avoid choppy markets.
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
# Works in both bull and bear markets via volatility-adjusted breakouts.

name = "4h_1d_donchian_breakout_with_volume_and_atr"
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
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) and ATR(50)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility filter: ATR(14) > 0.5 * ATR(50) (avoid low volatility/chop)
    vol_filter = atr14 > (0.5 * atr50)
    
    # Align volatility filter to 4h
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Calculate 20-period Donchian channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_filter_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above Donchian high + volume + volatility
        if close[i] > highest_high[i] and vol_confirm[i] and vol_filter_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below Donchian low + volume + volatility
        elif close[i] < lowest_low[i] and vol_confirm[i] and vol_filter_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses Donchian midpoint (mean reversion)
        elif position == 1 and close[i] <= donchian_mid[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= donchian_mid[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals