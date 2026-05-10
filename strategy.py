#!/usr/bin/env python3
"""
1h_4h_Donchian_1d_Volume_Regime
Hypothesis: Use 4h Donchian breakout for direction, 1d volume confirmation, and 1h volatility regime filter to avoid chop.
Designed for 15-30 trades/year, works in bull/bear via Donchian breakouts and volume filter.
"""

name = "1h_4h_Donchian_1d_Volume_Regime"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 1h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h volatility regime using ATR ratio (current ATR / 20-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma
    
    # Volatility filter: only trade when volatility is elevated (ATR ratio > 0.8) or extreme (< 1.2)
    # This avoids choppy low-volatility periods
    vol_filter = (atr_ratio > 0.8) & (atr_ratio < 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h Donchian (20), 1d volume MA (20), 1h ATR (14+20)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current 1d volume (need to align to 1h)
        # Get the 1d volume for the current day by finding the most recent completed 1d bar
        vol_1d_current = volume_1d[-1] if len(volume_1d) > 0 else 0
        
        if position == 0:
            # Long: price breaks above 4h Donchian high AND 1d volume above average AND volatility OK
            if (high[i] > donchian_high_aligned[i] and 
                volume > vol_ma_1d_aligned[i] * 1.5 and 
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low AND 1d volume above average AND volatility OK
            elif (low[i] < donchian_low_aligned[i] and 
                  volume > vol_ma_1d_aligned[i] * 1.5 and 
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low
            if low[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high
            if high[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals