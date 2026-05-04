#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation
# Uses 12h HMA21 for trend direction (bull/bear agnostic) and Donchian channels for breakout entries
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag on 4h timeframe
# Works in both bull and bear markets by following the 12h trend direction and using Donchian for structure
# Prioritizes BTC/ETH performance with SOL as secondary

name = "4h_Donchian20_12hHMA21_Trend_Volume"
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
    
    # Get 12h data for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21 for trend filter
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Calculate WMA for half period
    wma_half = np.full_like(close_12h, np.nan)
    for i in range(half_len, len(close_12h)):
        wma_half[i] = wma(close_12h[i-half_len+1:i+1], half_len)[-1]
    
    # Calculate WMA for full period
    wma_full = np.full_like(close_12h, np.nan)
    for i in range(21, len(close_12h)):
        wma_full[i] = wma(close_12h[i-21+1:i+1], 21)[-1]
    
    # Calculate HMA: 2*WMA(half) - WMA(full)
    hma_raw = 2 * wma_half - wma_full
    
    # Final WMA of sqrt(n) on the HMA raw
    hma_12h = np.full_like(close_12h, np.nan)
    for i in range(sqrt_len, len(hma_raw)):
        hma_12h[i] = wma(hma_raw[i-sqrt_len+1:i+1], sqrt_len)[-1]
    
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Donchian breakout with 12h HMA trend filter
        # Long: Price breaks above Donchian upper + volume spike + price above 12h HMA21 (uptrend)
        # Short: Price breaks below Donchian lower + volume spike + price below 12h HMA21 (downtrend)
        if position == 0:
            if (close[i] > highest_high[i] and volume_spike and 
                close[i] > hma_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lowest_low[i] and volume_spike and 
                  close[i] < hma_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian lower OR price below 12h HMA21 (trend change)
            if close[i] < lowest_low[i] or close[i] < hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian upper OR price above 12h HMA21 (trend change)
            if close[i] > highest_high[i] or close[i] > hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals