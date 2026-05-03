#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (HMA21) and volume confirmation.
# Long when price breaks above Donchian upper band in 12h uptrend (price > HMA21).
# Short when price breaks below Donchian lower band in 12h downtrend (price < HMA21).
# Volume must be > 1.5x 20-period MA to confirm breakout strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_12hHMA21_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21
    close_12h = df_12h['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA calculation
    wma_half = np.array([wma(close_12h[i:i+half_len], half_len) if i+half_len <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+21], 21) if i+21 <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    hma_21 = 2 * wma_half - wma_full
    hma_21_final = np.array([wma(hma_21[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(hma_21) else np.nan 
                             for i in range(len(hma_21))])
    
    # Pad to match length
    hma_21_padded = np.full(len(close_12h), np.nan)
    hma_21_padded[sqrt_len-1:-sqrt_len+1 if sqrt_len>1 else None] = hma_21_final[sqrt_len-1:]
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_padded)
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(hma_21_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > hma_21_aligned[i]   # 12h uptrend
        trend_down = close_val < hma_21_aligned[i]  # 12h downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above Donchian upper band AND 12h uptrend AND volume spike
            if close_val > highest_high[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND 12h downtrend AND volume spike
            elif close_val < lowest_low[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower band OR 12h trend turns down
            if close_val < lowest_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper band OR 12h trend turns up
            if close_val > highest_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals