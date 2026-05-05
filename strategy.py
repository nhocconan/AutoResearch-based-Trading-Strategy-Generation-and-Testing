#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA21 trend filter + volume spike confirmation
# Long when price breaks above Donchian(20) high AND price > 12h HMA21 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Donchian(20) low AND price < 12h HMA21 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses Donchian(20) midpoint OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian provides clear breakout signals in trending markets
# 12h HMA21 filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "4h_Donchian20_12hHMA21_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need enough for HMA21
        return np.zeros(n)
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h HMA21: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def calculate_wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        wma = np.convolve(values, weights, mode='valid') / weights.sum()
        result = np.full_like(values, np.nan)
        result[window-1:] = wma
        return result
    
    wma_10_12h = calculate_wma(close_12h, 10)
    wma_21_12h = calculate_wma(close_12h, 21)
    hma_12h_raw = 2 * wma_10_12h - wma_21_12h
    hma_12h = calculate_wma(hma_12h_raw, int(np.sqrt(21)) + 1)  # sqrt(21) ≈ 4.58 -> 5
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian(20) on primary timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high AND above 12h HMA21 AND volume confirmation
            if (close[i] > highest_high_20[i] and close[i] > hma_12h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low AND below 12h HMA21 AND volume confirmation
            elif (close[i] < lowest_low_20[i] and close[i] < hma_12h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals