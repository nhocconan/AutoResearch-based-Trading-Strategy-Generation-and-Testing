#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v2
# Hypothesis: 6h strategy using weekly pivot levels and Donchian breakouts with volume confirmation.
# Long: Price breaks above weekly H5 with volume > 1.8x 20-period average and close > open.
# Short: Price breaks below weekly L5 with volume > 1.8x 20-period average and close < open.
# Exit: Price returns to opposite weekly H3/L3 level.
# Uses 6h primary timeframe with 1w HTF for pivot/Donchian levels.
# Designed for low trade frequency (~10-25/year) to minimize fee drag while capturing major breaks.
# Weekly pivots provide strong institutional levels; Donchian confirms breakout strength.
# Works in bull markets via breakouts and bear markets via fade-from-extremes logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for weekly pivot and Donchian levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation (standard floor trader pivot)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla-like levels (H3/L3 for exit, H5/L5 for entry)
    h3_1w = pivot_1w + (range_1w * 1.1 / 4)
    l3_1w = pivot_1w - (range_1w * 1.1 / 4)
    h5_1w = pivot_1w + (range_1w * 1.1 * 2)  # Extended breakout level
    l5_1w = pivot_1w - (range_1w * 1.1 * 2)
    
    # Weekly Donchian channels (20-period)
    high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align all 1w levels to 6h
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    h5_1w_aligned = align_htf_to_ltf(prices, df_1w, h5_1w)
    l5_1w_aligned = align_htf_to_ltf(prices, df_1w, l5_1w)
    high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, high_20_1w)
    low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, low_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or
            np.isnan(h5_1w_aligned[i]) or np.isnan(l5_1w_aligned[i]) or
            np.isnan(high_20_1w_aligned[i]) or np.isnan(low_20_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        # Bullish candle: close > open
        bullish_candle = close[i] > open_prices[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to H3
            if close[i] <= h3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3
            if close[i] >= l3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly H5 AND 20-period high with volume and bullish candle
            if (close[i] > h5_1w_aligned[i] and
                close[i] > high_20_1w_aligned[i] and
                volume_confirmed and
                bullish_candle):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly L5 AND 20-period low with volume and bearish candle
            elif (close[i] < l5_1w_aligned[i] and
                  close[i] < low_20_1w_aligned[i] and
                  volume_confirmed and
                  bearish_candle):
                position = -1
                signals[i] = -0.25
    
    return signals