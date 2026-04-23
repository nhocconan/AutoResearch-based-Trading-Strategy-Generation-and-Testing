#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based trend filter and volume confirmation.
Long when price breaks above Donchian upper band and ATR(1d) > ATR(20d) (strong volatility regime) with volume > 1.5x average.
Short when price breaks below Donchian lower band and ATR(1d) > ATR(20d) with volume > 1.5x average.
Exit on opposite Donchian break or volatility contraction (ATR(1d) < ATR(20d)).
Uses 4h timeframe targeting 75-200 total trades over 4 years.
Donchian channels provide clear breakout levels, 1d ATR ratio filters for explosive moves, volume confirms strength.
Designed to capture strong momentum bursts in both bull and bear markets while avoiding choppy periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR-based trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(1d) and ATR(20d) on daily timeframe
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr1 = pd.Series(tr1).ewm(span=1, adjust=False, min_periods=1).mean().values  # ATR(1) for smoothing
    atr1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values  # ATR(14) ~ 1d ATR
    
    # ATR(20d) - longer term volatility
    tr20 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr20 = np.maximum(tr20, np.abs(low_1d[1:] - close_1d[:-1]))
    tr20 = np.concatenate([[np.nan], tr20])
    atr20d = pd.Series(tr20).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d ATR indicators to 4h timeframe
    atr1d_aligned = align_htf_to_ltf(prices, df_1d, atr1d)
    atr20d_aligned = align_htf_to_ltf(prices, df_1d, atr20d)
    
    # Donchian(20) on 4h timeframe
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(atr1d_aligned[i]) or np.isnan(atr20d_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr1d_val = atr1d_aligned[i]
        atr20d_val = atr20d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Volatility regime filter: ATR(1d) > ATR(20d) indicates expanding volatility
        vol_regime = atr1d_val > atr20d_val
        
        if position == 0:
            # Long: price breaks above Donchian upper AND volatility expanding AND volume > 1.5x average
            if (price > upper[i] and vol_regime and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND volatility expanding AND volume > 1.5x average
            elif (price < lower[i] and vol_regime and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower OR volatility contraction
                if (price < lower[i] or not vol_regime):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper OR volatility contraction
                if (price > upper[i] or not vol_regime):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0