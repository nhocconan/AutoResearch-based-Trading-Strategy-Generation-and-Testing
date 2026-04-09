#!/usr/bin/env python3
# 4h_donchian_1d_camarilla_volume_chop_v3
# Hypothesis: 4h Donchian(20) breakout with 1d Camarilla H4/L4 filter, volume confirmation, and chop regime filter.
# Uses 4h timeframe for optimal trade frequency (~30-60 trades/year). Donchian provides trend following breakouts,
# Camarilla H4/L4 from 1d acts as strong bias filter (only trade in direction of daily pivot extremes),
# volume spike confirms institutional interest, chop filter avoids whipsaws in ranging markets.
# Designed to work in bull/bear markets: breakouts capture trends, Camarilla filter avoids counter-trend fakes,
# chop filter reduces losses in sideways markets. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_camarilla_volume_chop_v3"
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
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 4h timeframe (completed 4h candle only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Get 1d HTF data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels for daily (H4/L4 for stronger direction filter)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (completed daily candle only)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    # Chop regime filter using 4h data (avoid whipsaws in ranging markets)
    chop_window = 14
    if len(df_4h) >= chop_window:
        high_4h_series = pd.Series(df_4h['high'])
        low_4h_series = pd.Series(df_4h['low'])
        true_range = np.maximum(
            high_4h_series - low_4h_series,
            np.maximum(
                np.abs(high_4h_series - df_4h['close']),
                np.abs(low_4h_series - df_4h['close'])
            )
        )
        atr = true_range.rolling(window=chop_window, min_periods=chop_window).mean().values
        highest_high = high_4h_series.rolling(window=chop_window, min_periods=chop_window).max().values
        lowest_low = low_4h_series.rolling(window=chop_window, min_periods=chop_window).min().values
        chop = 100 * np.log10((highest_high - lowest_low) / (atr * chop_window)) / np.log10(chop_window)
        chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
        chop_filter = chop_aligned > 50  # Only trade when chop > 50 (avoid strong trends for mean reversion, but we use it as trend filter)
    else:
        chop_filter = np.ones(n, dtype=bool)  # Default to true if not enough data
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian lower band
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian upper band
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 4h Donchian upper, above 1d H4, with volume spike, and chop > 50
            if (close[i] > donchian_upper_aligned[i]) and (close[i] > h4_1d_aligned[i]) and vol_spike[i] and chop_filter[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h Donchian lower, below 1d L4, with volume spike, and chop > 50
            elif (close[i] < donchian_lower_aligned[i]) and (close[i] < l4_1d_aligned[i]) and vol_spike[i] and chop_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals