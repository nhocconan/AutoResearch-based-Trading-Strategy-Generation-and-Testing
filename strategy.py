#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
# Donchian breakout captures momentum in both bull/bear markets
# 1w HMA(21) filters for higher timeframe trend direction (avoid counter-trend trades)
# Volume confirmation: current volume > 2.0x 20-period MA to ensure conviction
# Entry: Long when price > Donchian Upper AND price > 1w HMA AND volume spike
# Entry: Short when price < Donchian Lower AND price < 1w HMA AND volume spike
# Exit: When price crosses Donchian midpoint (mean reversion to average)
# Uses discrete sizing (0.25) to minimize fee churn. Target: 20-80 trades over 4 years.

name = "1d_Donchian20_1wHMA21_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for HMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate 1w HMA(21)
    close_1w = df_1w['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA components for HMA
    wma_half = np.array([np.nan] * half_len + list(wma(close_1w, half_len)))
    wma_full = np.array([np.nan] * 21 + list(wma(close_1w, 21)))
    hma_2wma = 2 * wma_half - wma_full
    hma_21 = np.array([np.nan] * (21 + half_len - 1) + list(wma(hma_2wma[~np.isnan(hma_2wma)], sqrt_len)))
    
    # Pad hma_21 to match close_1w length
    if len(hma_21) < len(close_1w):
        hma_21 = np.concatenate([hma_21, np.full(len(close_1w) - len(hma_21), np.nan)])
    elif len(hma_21) > len(close_1w):
        hma_21 = hma_21[:len(close_1w)]
    
    # Align 1w HMA to 1d
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Calculate 1d Donchian(20)
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian Upper AND above 1w HMA AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > hma_21_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian Lower AND below 1w HMA AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < hma_21_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals