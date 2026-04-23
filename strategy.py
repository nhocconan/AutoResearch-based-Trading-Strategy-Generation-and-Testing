#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume spike confirmation.
Long when price breaks above Donchian upper band AND 1d ATR(14) > 1.5x 50-period average AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND 1d ATR(14) > 1.5x 50-period average AND volume > 2.0x 20-period average.
Exit when price crosses Donchian middle band (20-period SMA).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
ATR filter ensures we only trade during sufficient volatility regimes, reducing whipsaw in low-volatility bear markets.
"""

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
    
    # Load 1d data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 50-period average of ATR for regime filter
    atr_ma_50 = pd.Series(atr14_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR and its MA to 4h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Calculate Donchian channels on 4h timeframe (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper band (20-period high)
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian middle band (20-period SMA of close)
    middle_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels to 4h timeframe (primary)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(middle_4h_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr_ma_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr14_1d_aligned[i]
        atr_ma_val = atr_ma_50_aligned[i]
        
        # Volatility filter: only trade when current ATR > 1.5x its 50-period average
        vol_filter = atr_val > 1.5 * atr_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND volatility filter AND volume spike
            if (price > upper_4h_aligned[i] and 
                vol_filter and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND volatility filter AND volume spike
            elif (price < lower_4h_aligned[i] and 
                  vol_filter and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian middle band
            if position == 1 and price < middle_4h_aligned[i]:
                exit_signal = True
            elif position == -1 and price > middle_4h_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATRVolFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0