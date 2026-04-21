#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above upper Donchian channel and volume > 1.5x average with ADX > 25.
Short when price breaks below lower Donchian channel and volume > 1.5x average with ADX > 25.
Exit when price returns to middle of Donchian channel or volume drops below average.
Designed for ~25-35 trades/year to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily high, low, close for ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if ADX not ready
        if np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels for lookback period 20
        if i >= 20:
            high_20 = np.max(prices['high'].iloc[i-19:i+1].values)
            low_20 = np.min(prices['low'].iloc[i-19:i+1].values)
            mid_20 = (high_20 + low_20) / 2
        else:
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, volume surge, ADX > 25
            if (price_close > high_20 and 
                vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, volume surge, ADX > 25
            elif (price_close < low_20 and 
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to middle of Donchian or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= middle or volume < average
                if (price_close <= mid_20 or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= middle or volume < average
                if (price_close >= mid_20 or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume1.5x_ADX25_Trend"
timeframe = "4h"
leverage = 1.0