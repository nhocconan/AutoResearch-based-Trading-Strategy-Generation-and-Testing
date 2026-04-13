#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h ATR-based volatility filter + volume confirmation
    # Long when: price breaks above Donchian(20) high AND 12h ATR ratio > 1.2 (expanding volatility) AND volume > 1.3x avg volume
    # Short when: price breaks below Donchian(20) low AND 12h ATR ratio > 1.2 AND volume > 1.3x avg volume
    # Exit when: price crosses Donchian midpoint OR ATR ratio drops below 0.8 (low volatility)
    # Uses discrete sizing (0.25) targeting 50-100 trades over 4 years.
    # Works in bull/bear via volatility expansion capturing momentum bursts after consolidation.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ATR calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(14)
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align indices
    
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h ATR(50) for longer-term average
    tr_12h_long = np.concatenate([[np.nan], tr_12h])
    atr_12h_long = pd.Series(tr_12h_long).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR ratio: short-term / long-term ( >1 = expanding volatility)
    atr_ratio = np.where(atr_12h_long > 0, atr_12h / atr_12h_long, np.nan)
    
    # Align 12h ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # Calculate Donchian(20) channels on 4h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Volatility expansion filter (ATR ratio > 1.2)
        vol_expansion = atr_ratio_aligned[i] > 1.2
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_expansion and vol_ok and position != 1
        short_entry = short_breakout and vol_expansion and vol_ok and position != -1
        
        # Exit conditions: price crosses Donchian midpoint OR ATR ratio drops below 0.8 (low volatility)
        exit_long = close[i] < donchian_mid[i] or atr_ratio_aligned[i] < 0.8
        exit_short = close[i] > donchian_mid[i] or atr_ratio_aligned[i] < 0.8
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0