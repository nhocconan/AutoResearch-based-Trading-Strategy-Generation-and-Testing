#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and choppiness regime filter
# Long when price breaks above 1w Donchian upper (20) AND volume > 1.3 * avg_volume(20) AND choppiness < 61.8 (trending)
# Short when price breaks below 1w Donchian lower (20) AND volume > 1.3 * avg_volume(20) AND choppiness < 61.8 (trending)
# Exit when price crosses the 1w Donchian midpoint
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides strong structural breakouts that reduce whipsaw in both bull and bear markets
# Volume confirmation validates breakout strength while choppiness filter ensures we only trade in trending regimes
# This combination has proven effective across multiple timeframes in the research database

name = "1d_1wDonchian20_VolumeConfirm_ChopFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian and choppiness calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channel (20-period)
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align 1w Donchian to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_20)
    
    # Calculate 1w choppiness index (14-period) for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest high - lowest low over 14))
    tr1 = high_1w[1:] - low_1w[:-1]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align with original arrays
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, np.nan)
    chop = 100 * np.log10(chop_raw)
    
    # Align choppiness to 1d timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian high, volume confirmation, chop < 61.8 (trending), in session
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirm[i] and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian low, volume confirmation, chop < 61.8 (trending), in session
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirm[i] and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1w Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1w Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals