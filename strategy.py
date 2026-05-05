#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout + 12h HMA21 trend filter + volume spike confirmation
# Long when price breaks above 12h Donchian upper channel AND price > 12h HMA21 AND volume > 2.0 * avg_volume(20) on 4h
# Short when price breaks below 12h Donchian lower channel AND price < 12h HMA21 AND volume > 2.0 * avg_volume(20) on 4h
# Exit when price crosses back through 12h Donchian midpoint OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Donchian provides structure from higher timeframe reducing false breakouts
# 12h HMA21 filters for primary trend alignment to avoid counter-trend trades
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
    
    # Get 12h data ONCE before loop for Donchian and HMA calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need enough for HMA21 and Donchian20
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Align 12h Donchian channels to 4h timeframe (wait for completed 12h bar)
    highest_high_20_aligned = align_htf_to_ltf(prices, df_12h, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Calculate 12h HMA21 (Hull Moving Average)
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate WMA for half period
    half_period = 21 // 2
    if len(close_12h) >= half_period:
        wma_half = np.array([wma(close_12h[i:i+half_period], half_period) 
                            for i in range(len(close_12h) - half_period + 1)])
        # Pad the beginning with NaN to align with original array
        wma_half_padded = np.full(len(close_12h), np.nan)
        wma_half_padded[half_period-1:] = wma_half
    else:
        wma_half_padded = np.full(len(close_12h), np.nan)
    
    # Calculate WMA for full period
    if len(close_12h) >= 21:
        wma_full = np.array([wma(close_12h[i:i+21], 21) 
                            for i in range(len(close_12h) - 21 + 1)])
        # Pad the beginning with NaN to align with original array
        wma_full_padded = np.full(len(close_12h), np.nan)
        wma_full_padded[20:] = wma_full
    else:
        wma_full_padded = np.full(len(close_12h), np.nan)
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half_padded - wma_full_padded
    
    # Calculate final HMA: WMA of raw_hma with period = sqrt(21) ≈ 4.58 -> round to 5
    sqrt_n = int(np.sqrt(21))  # 4
    if sqrt_n < 1:
        sqrt_n = 1
    if len(raw_hma) >= sqrt_n and not np.all(np.isnan(raw_hma)):
        # WMA of raw_hma
        hma_values = np.array([wma(raw_hma[i:i+sqrt_n], sqrt_n) 
                              for i in range(len(raw_hma) - sqrt_n + 1)])
        # Pad the beginning with NaN to align with original array
        hma_21_padded = np.full(len(raw_hma), np.nan)
        hma_21_padded[sqrt_n-1:] = hma_values
        hma_21 = hma_21_padded
    else:
        hma_21 = np.full(len(close_12h), np.nan)
    
    # Align 12h HMA21 to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper channel, above 12h HMA21, volume confirmation, in session
            if (close[i] > highest_high_20_aligned[i] and close[i-1] <= highest_high_20_aligned[i-1] and 
                close[i] > hma_21_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower channel, below 12h HMA21, volume confirmation, in session
            elif (close[i] < lowest_low_20_aligned[i] and close[i-1] >= lowest_low_20_aligned[i-1] and 
                  close[i] < hma_21_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 12h Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals