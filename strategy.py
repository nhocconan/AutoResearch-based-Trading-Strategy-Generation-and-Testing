#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume spike confirmation.
- Long when price breaks above Donchian upper (20-bar high) AND 12h HMA21 slope > 0 (bullish trend)
- Short when price breaks below Donchian lower (20-bar low) AND 12h HMA21 slope < 0 (bearish trend)
- Volume must be > 2.0 * median volume of last 50 bars (strong volume confirmation to avoid fakeouts)
- Exit on opposite Donchian breakout or trend reversal (12h HMA21 slope crosses zero)
- Uses 4h primary timeframe with 12h HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian channels provide clear structure for breakouts in both trending and ranging markets
- 12h HMA21 smooths trend and reduces whipsaws vs EMA/SMA
- Volume spike filter ensures breakouts have conviction, reducing false signals
- Designed for BTC/ETH with edge in breakout continuation during trends and avoidance of chop
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-bar high/low)
    # Upper = max(high of last 20 bars), Lower = min(low of last 20 bars)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data ONCE before loop for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21 (Hull Moving Average)
    close_12h = df_12h['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function for HMA calculation
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    wma_diff = 2 * wma_half - wma_full
    # Pad the beginning with NaN since convolution reduces length
    wma_diff_padded = np.full(len(close_12h), np.nan)
    wma_diff_padded[half_len-1:len(wma_diff)+half_len-1] = wma_diff
    hma_21_12h = wma(wma_diff_padded, sqrt_len)
    # Pad the beginning again for final WMA
    hma_21_padded = np.full(len(close_12h), np.nan)
    hma_21_padded[sqrt_len-1:len(hma_21_12h)+sqrt_len-1] = hma_21_12h
    
    # Align 12h HMA21 to 4h timeframe
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_padded)
    
    # Calculate HMA21 slope (trend direction) - positive slope = bullish
    hma_slope = np.diff(hma_21_12h_aligned, prepend=np.nan)
    
    # Volume confirmation: volume > 2.0 * median volume of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_slope[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, bullish trend (HMA slope > 0), volume confirmation
            if close[i] > donchian_upper[i] and hma_slope[i] > 0 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, bearish trend (HMA slope < 0), volume confirmation
            elif close[i] < donchian_lower[i] and hma_slope[i] < 0 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR trend reversal (HMA slope < 0)
            if close[i] < donchian_lower[i] or hma_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR trend reversal (HMA slope > 0)
            if close[i] > donchian_upper[i] or hma_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0