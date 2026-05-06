#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakout with volume confirmation and choppiness regime filter
# Long when price breaks above weekly Donchian upper channel AND volume > 1.5 * avg_volume(20) AND weekly chop < 61.8 (trending)
# Short when price breaks below weekly Donchian lower channel AND volume > 1.5 * avg_volume(20) AND weekly chop < 61.8 (trending)
# Exit when price crosses weekly Donchian midpoint (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Donchian provides strong structural support/resistance levels
# Volume confirmation ensures breakout validity while limiting overtrading
# Chop filter avoids false breakouts in ranging markets
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
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align weekly Donchian to 1d timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate weekly choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(HH - LL) * 14)) / log10(14)
    tr1 = high_1w[1:] - low_1w[:-1]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(hh_14 - ll_14) * 14
    chop_denom = np.where(chop_denom == 0, np.nan, chop_denom)
    chop_raw = 100 * np.log10(sum_atr_14) / chop_denom / np.log10(14)
    chop_14 = np.where(np.isnan(chop_raw) | (hh_14 - ll_14) == 0, 50, chop_raw)
    
    # Align weekly chop to 1d timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_14)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper, volume confirm, chop < 61.8 (trending)
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirm[i] and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower, volume confirm, chop < 61.8 (trending)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirm[i] and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals