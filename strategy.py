#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_regime_v1
# Hypothesis: Daily strategy using weekly Donchian channel breakouts with volume confirmation and choppiness regime filter.
# Long: Price breaks above weekly Donchian high (20) with volume > 1.5x 20-day average AND weekly chop > 61.8 (ranging market)
# Short: Price breaks below weekly Donchian low (20) with volume > 1.5x 20-day average AND weekly chop > 61.8 (ranging market)
# Exit: Price returns to weekly Donchian midpoint or opposite breakout
# Uses 1d primary timeframe with 1w HTF for Donchian channels and choppiness filter.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Donchian channels and choppiness
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    def calculate_donchian(high_arr, low_arr, period=20):
        upper = np.full_like(high_arr, np.nan)
        lower = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-period+1:i+1])
            lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    donchian_high_1w, donchian_low_1w = calculate_donchian(high_1w, low_1w, 20)
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2.0
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    donchian_mid_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    
    # Calculate weekly choppiness index (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr_sum = np.zeros_like(close_arr)
        true_range = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr = max(high_arr[i] - low_arr[i], 
                     abs(high_arr[i] - close_arr[i-1]),
                     abs(low_arr[i] - close_arr[i-1]))
            true_range[i] = tr
        # Calculate ATR using Wilder's smoothing (equivalent to RMA)
        atr = np.zeros_like(close_arr)
        atr[period] = np.mean(true_range[1:period+1]) if period < len(true_range) else 0
        for i in range(period+1, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        # Sum ATR over period
        atr_sum = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Calculate choppiness
        chop = np.zeros_like(close_arr)
        max_high = np.zeros_like(close_arr)
        min_low = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            max_high[i] = np.max(high_arr[i-period+1:i+1])
            min_low[i] = np.min(low_arr[i-period+1:i+1])
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    chop_1w = calculate_chop(high_1w, low_1w, close_1w, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or 
            np.isnan(donchian_mid_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop_1w_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price returns to weekly midpoint or breaks below weekly low (opposite signal)
            if close[i] <= donchian_mid_1w_aligned[i] or close[i] < donchian_low_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to weekly midpoint or breaks above weekly high (opposite signal)
            if close[i] >= donchian_mid_1w_aligned[i] or close[i] > donchian_high_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly high with volume confirmation in ranging market
            if close[i] > donchian_high_1w_aligned[i] and volume_confirmed and chop_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly low with volume confirmation in ranging market
            elif close[i] < donchian_low_1w_aligned[i] and volume_confirmed and chop_filter:
                position = -1
                signals[i] = -0.25
    
    return signals