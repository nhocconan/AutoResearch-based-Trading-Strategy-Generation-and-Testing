#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_spike_v10
# Hypothesis: 4h strategy using 1d Camarilla levels with volume confirmation and chop regime filter.
# Long: Price breaks above H4 with volume > 1.5x 20-period average and CHOP > 61.8 (range regime).
# Short: Price breaks below L4 with volume > 1.5x 20-period average and CHOP > 61.8.
# Exit: Price returns to opposite Camarilla level (H3 for longs, L3 for shorts).
# Uses 4h primary timeframe with 1d HTF for Camarilla levels and chop filter.
# Designed for low trade frequency (~20-50/year) to minimize fee drag while capturing institutional breakouts in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_spike_v10"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla levels and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Choppiness Index (CHOP) - 14 period
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        tr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]),
                       abs(low_arr[i] - close_arr[i-1]))
        # Smooth TR with Wilder's smoothing (alpha = 1/period)
        atr[period-1] = np.nanmean(tr[1:period]) if period > 1 else tr[1]
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # CHOP formula
        sum_tr = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
        highest_high = np.zeros_like(close_arr)
        lowest_low = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            highest_high[i] = np.max(high_arr[i-period+1:i+1])
            lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        chop = np.full_like(close_arr, 50.0, dtype=float)
        for i in range(period-1, len(close_arr)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(sum_tr[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Align 1d Camarilla levels to 4h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Chop regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion at extremes)
        chop_filter = chop_1d_aligned[i] > 61.8
        # Bullish candle: close > open
        bullish_candle = close[i] > open_prices[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to H3
            if close[i] <= h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3
            if close[i] >= l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H4 with volume and bullish candle in ranging market
            if (close[i] > h4_1d_aligned[i] and    # Break above H4
                volume_confirmed and               # Volume spike
                bullish_candle and                 # Bullish candle
                chop_filter):                      # Ranging market (CHOP > 61.8)
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L4 with volume and bearish candle in ranging market
            elif (close[i] < l4_1d_aligned[i] and  # Break below L4
                  volume_confirmed and             # Volume spike
                  bearish_candle and               # Bearish candle
                  chop_filter):                    # Ranging market (CHOP > 61.8)
                position = -1
                signals[i] = -0.25
    
    return signals