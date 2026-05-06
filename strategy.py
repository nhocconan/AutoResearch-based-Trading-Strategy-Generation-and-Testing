#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with 1d volume confirmation and chop regime filter
# Long when price breaks above 1w Donchian(20) high AND 1d volume > 1.5 * avg_volume(20) AND chop < 61.8 (trending)
# Short when price breaks below 1w Donchian(20) low AND 1d volume > 1.5 * avg_volume(20) AND chop < 61.8 (trending)
# Exit when price crosses 1w Donchian(20) midpoint
# Uses discrete sizing 0.30 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Donchian provides strong structural breaks that work in both bull and bear markets
# Volume confirmation ensures breakout validity while chop filter avoids false signals in ranging markets
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns)

name = "1d_1wDonchian20_Breakout_Volume_ChopFilter"
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
    
    # Get 1w data ONCE before loop for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels: 20-period high/low
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Align 1w Donchian to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get 1d data ONCE before loop for chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed 1d bars for chop
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Chopiness Index: 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14)
    # Simplified version: 100 * log10(ATR_sum / (max_high - min_low)) / log10(period)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum) / np.log10(14) / np.log10(highest_high_14 - lowest_low_14)
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop_raw)  # Default to neutral when range=0
    
    # Align 1d chop to 1d timeframe (no delay needed as it's already LTF)
    chop_aligned = chop  # Already on 1d timeframe
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian high, volume confirmation, chop < 61.8 (trending), in session
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirm[i] and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1w Donchian low, volume confirmation, chop < 61.8 (trending), in session
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirm[i] and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 1w Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals