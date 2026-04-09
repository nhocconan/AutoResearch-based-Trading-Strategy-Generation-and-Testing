#!/usr/bin/env python3
# 12h_weekly_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategy using 1w Camarilla pivot levels with volume confirmation and chop regime filter.
# Long: Price breaks above weekly H4 with volume > 1.8x 30-period average and CHOP > 61.8 (range regime).
# Short: Price breaks below weekly L4 with volume > 1.8x 30-period average and CHOP > 61.8 (range regime).
# Exit: Price returns to opposite Camarilla level (H3 for longs, L3 for shorts).
# Uses 1w timeframe for structure, 12h for execution. Designed for low frequency (12-37 trades/year) to minimize fee drag.
# Works in bull markets via breakouts and bear markets via fade-from-extremes logic in range regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
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
    
    # Volume average for confirmation (30-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=30, min_periods=30).mean().values
    
    # Choppiness Index (CHOP) regime filter
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # CHOP = 100 * log10(sum(TR,14) / (HHV(high,14) - LLV(low,14))) / log10(14)
    sum_tr = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low = low_s.rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * (np.log10(sum_tr) - np.log10(highest_high - lowest_low)) / np.log10(14)
    chop_values = np.nan_to_num(chop_raw, nan=50.0)  # Replace NaN with neutral 50
    
    # Get 1w data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    h3_1w = pivot_1w + (range_1w * 1.1 / 4)
    l3_1w = pivot_1w - (range_1w * 1.1 / 4)
    h4_1w = pivot_1w + (range_1w * 1.1 / 2)
    l4_1w = pivot_1w - (range_1w * 1.1 / 2)
    
    # Align 1w Camarilla levels to 12h
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    h4_1w_aligned = align_htf_to_ltf(prices, df_1w, h4_1w)
    l4_1w_aligned = align_htf_to_ltf(prices, df_1w, l4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or
            np.isnan(h4_1w_aligned[i]) or np.isnan(l4_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_values[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 30-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        # Regime filter: CHOP > 61.8 indicates range-bound market (fade extremes)
        regime_filter = chop_values[i] > 61.8
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
            # Long entry: Price breaks above H4 with volume, regime filter, and bullish candle
            if (close[i] > h4_1w_aligned[i] and    # Break above H4
                volume_confirmed and               # Volume spike
                regime_filter and                  # Range regime (CHOP > 61.8)
                bullish_candle):                   # Bullish candle
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L4 with volume, regime filter, and bearish candle
            elif (close[i] < l4_1w_aligned[i] and  # Break below L4
                  volume_confirmed and             # Volume spike
                  regime_filter and                # Range regime (CHOP > 61.8)
                  bearish_candle):                 # Bearish candle
                position = -1
                signals[i] = -0.25
    
    return signals