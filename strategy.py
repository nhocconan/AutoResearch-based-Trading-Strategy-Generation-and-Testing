#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume spike filter.
Long when price breaks above Donchian upper band AND close > 1d HMA21 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d HMA21 AND volume > 1.8x 20-period average.
Exit when price reverts to Donchian midpoint or ATR-based stoploss (2.0x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
Donchian channels provide clear trend structure, while 1d HMA21 filters for higher-timeframe trend alignment.
Volume spike confirms breakout validity. Works in both bull and bear markets by avoiding counter-trend breakouts.
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
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    # Upper band = max(high, 20)
    # Lower band = min(low, 20)
    # Middle band = (upper + lower) / 2
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_4h = (upper_4h + lower_4h) / 2.0
    
    # Align 4h Donchian levels to 4h timeframe (shift by 1 to use previous bar's levels)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Load 1d data for HMA21 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on 1d data
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA for half period
    wma_half = np.full(len(close_1d), np.nan)
    for i in range(half_len - 1, len(close_1d)):
        wma_half[i] = np.mean(close_1d[i - half_len + 1:i + 1] * np.arange(1, half_len + 1))
    
    # Calculate WMA for full period
    wma_full = np.full(len(close_1d), np.nan)
    for i in range(21 - 1, len(close_1d)):
        wma_full[i] = np.mean(close_1d[i - 21 + 1:i + 1] * np.arange(1, 22))
    
    # Calculate raw HMA
    raw_hma = 2 * wma_half - wma_full
    hma_1d = np.full(len(close_1d), np.nan)
    for i in range(sqrt_len - 1, len(raw_hma)):
        if not np.isnan(raw_hma[i]):
            hma_1d[i] = np.mean(raw_hma[i - sqrt_len + 1:i + 1] * np.arange(1, sqrt_len + 1))
    
    # Align 1d HMA21 to 4h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(middle_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper band AND close > 1d HMA21 AND volume spike
            if (price > upper_4h_aligned[i] and 
                close[i] > hma_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band AND close < 1d HMA21 AND volume spike
            elif (price < lower_4h_aligned[i] and 
                  close[i] < hma_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle band or ATR stoploss
                if price <= middle_4h_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle band or ATR stoploss
                if price >= middle_4h_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dHMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0