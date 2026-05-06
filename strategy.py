#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper(20) AND 12h HMA(21) rising AND volume > 1.3 * avg_volume(20) on 4h
# Short when price breaks below 4h Donchian lower(20) AND 12h HMA(21) falling AND volume > 1.3 * avg_volume(20) on 4h
# Exit when price crosses back through the 4h Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels provide structural breakout levels that reduce whipsaw
# 12h HMA(21) trend filter ensures we trade with the dominant intermediate trend
# Volume confirmation (1.3x) validates breakout strength while limiting overtrading

name = "4h_Donchian20_12hHMA21_Trend_VolumeConfirm"
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
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars for Donchian
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_roll_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = high_roll_20
    donchian_lower_4h = low_roll_20
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Align 4h Donchian to 4h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Get 12h data ONCE before loop for HMA(21) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need at least 21 completed 12h bars for HMA
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, window):
        half = window // 2
        sqrt_n = int(np.sqrt(window))
        wma_half = wma(values, half)
        wma_full = wma(values, window)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_n)
    
    # Pad HMA to match original length
    hma_12h_raw = hma(close_12h, 21)
    hma_12h = np.full_like(close_12h, np.nan)
    hma_12h[20:] = hma_12h_raw  # HMA(21) needs 21 bars to start
    
    # Align 12h HMA to 4h timeframe (wait for completed 12h bar)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper, 12h HMA rising, volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                hma_12h_aligned[i] > hma_12h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian lower, 12h HMA falling, volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  hma_12h_aligned[i] < hma_12h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 4h Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 4h Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals