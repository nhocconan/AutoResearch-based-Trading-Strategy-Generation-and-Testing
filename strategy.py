#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakout with volume confirmation and choppiness regime filter
# Long when price breaks above weekly Donchian high(20) AND volume > 1.5 * avg_volume(20) AND weekly chop < 61.8 (trending)
# Short when price breaks below weekly Donchian low(20) AND volume > 1.5 * avg_volume(20) AND weekly chop < 61.8 (trending)
# Exit when price crosses weekly Donchian midpoint (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Donchian breakouts capture strong trending moves with low frequency
# Volume confirmation ensures breakout validity while reducing false signals
# Choppiness filter avoids ranging markets where breakouts fail
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
    
    # Get weekly data ONCE before loop for Donchian and chop calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (highest_high_20 + lowest_low_20) / 2.0
    
    # Align weekly Donchian to daily timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_20)
    
    # Calculate weekly choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR(1),14) / (log10(HH - LL) * 14)) / log10(14)
    tr1 = np.maximum(high_1w[1:] - low_1w[:-1], np.absolute(high_1w[1:] - close_1w[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with high/low arrays
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh_ll_14 = pd.Series(highest_high_20 - lowest_low_20).rolling(window=14, min_periods=14).max().values
    chop_1w = 100 * np.log10(sum_atr1 / (hh_ll_14 * 14)) / np.log10(14)
    chop_1w = np.where(hh_ll_14 == 0, 50, chop_1w)  # avoid division by zero
    chop_1w = np.where(np.isnan(chop_1w), 50, chop_1w)  # fill NaN with neutral
    
    # Align weekly chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high, volume spike, trending regime (chop < 61.8)
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirm[i] and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low, volume spike, trending regime (chop < 61.8)
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirm[i] and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midpoint (mean reversion)
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midpoint (mean reversion)
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals