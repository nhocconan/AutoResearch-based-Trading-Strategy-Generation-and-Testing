#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation
# Donchian breakouts capture momentum. HMA21 on 12h filters trend direction. Volume spike confirms strength.
# Designed for 20-50 trades/year on 4h to minimize fee drag. Works in bull markets via breakouts and
# in bear markets via short breakdowns. Uses discrete position sizing (0.0, ±0.30) to reduce churn.

name = "4h_Donchian20_12hHMA21_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA21 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h data
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = np.array([wma(close_12h[i:i+half_len], half_len) 
                         if i+half_len <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+21], 21) 
                         if i+21 <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    raw_hma = 2 * wma_half - wma_full
    hma_21 = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) 
                       if i+sqrt_len <= len(raw_hma) else np.nan 
                       for i in range(len(raw_hma))])
    
    # Align HMA21 to 4h timeframe (wait for completed 12h bar)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Calculate Donchian(20) channels on 4h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(hma_21_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 12h HMA21 uptrend AND volume spike
            if (close[i] > highest_high_20[i] and 
                close[i] > hma_21_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower AND 12h HMA21 downtrend AND volume spike
            elif (close[i] < lowest_low_20[i] and 
                  close[i] < hma_21_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR trend reverses
            if close[i] < lowest_low_20[i] or close[i] < hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR trend reverses
            if close[i] > highest_high_20[i] or close[i] > hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals