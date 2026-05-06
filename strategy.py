#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and choppiness regime filter
# Long when price breaks above 1w Donchian upper channel AND choppiness index < 38.2 (trending) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1w Donchian lower channel AND choppiness index < 38.2 (trending) AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back through 1w Donchian middle channel (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian breakouts capture strong trending moves while avoiding false breakouts in ranging markets
# Choppiness filter ensures we only trade in trending regimes (CHOP < 38.2) and avoid whipsaws in ranging markets
# Volume confirmation validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "1d_1wDonchian20_Breakout_Volume_Chop"
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
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    middle_channel_20 = (highest_high_20 + lowest_low_20) / 2.0
    
    # Align 1w Donchian channels to 1d timeframe (wait for completed 1w bar)
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    middle_channel_20_aligned = align_htf_to_ltf(prices, df_1w, middle_channel_20)
    
    # Calculate 1w Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * (HHV - LLV))) / log10(n)
    tr1 = np.maximum(
        np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1])),
        np.abs(low_1w[1:] - close_1w[:-1])
    )
    tr1 = np.concatenate([[np.nan], tr1])  # align with close_1w indices
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hhllv = highest_high_20 - lowest_low_20
    chop_denominator = 14 * np.log10(hhllv)
    chop_numerator = np.log10(sum_atr1)
    chop = 100 * (chop_numerator / chop_denominator)
    # Handle division by zero and invalid values
    chop = np.where((hhllv == 0) | (chop_denominator == 0) | np.isnan(chop) | np.isinf(chop), 50, chop)
    
    # Align 1w Choppiness Index to 1d timeframe (wait for completed 1w bar)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or 
            np.isnan(middle_channel_20_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper channel, trending regime (CHOP < 38.2), volume spike
            if (close[i] > highest_high_20_aligned[i] and 
                chop_aligned[i] < 38.2 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower channel, trending regime (CHOP < 38.2), volume spike
            elif (close[i] < lowest_low_20_aligned[i] and 
                  chop_aligned[i] < 38.2 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Donchian middle channel (mean reversion)
            if close[i] < middle_channel_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Donchian middle channel (mean reversion)
            if close[i] > middle_channel_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals