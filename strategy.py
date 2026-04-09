#!/usr/bin/env python3
# 1d_weekly_camarilla_pivot_volume_spike_v2
# Hypothesis: Daily strategy using 1w Camarilla pivot levels with volume confirmation.
# Long: Price breaks above weekly H4 with volume > 2.0x 20-period average and close > open.
# Short: Price breaks below weekly L4 with volume > 2.0x 20-period average and close < open.
# Exit: Price returns to opposite Camarilla level (H3 for longs, L3 for shorts).
# Uses 1d primary timeframe with 1w HTF for Camarilla levels.
# Designed for low trade frequency (~10-25/year) to minimize fee drag and work in both bull and bear markets via breakouts and fade-from-extremes logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_camarilla_pivot_volume_spike_v2"
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
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
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
    
    # Align 1w Camarilla levels to 1d
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    h4_1w_aligned = align_htf_to_ltf(prices, df_1w, h4_1w)
    l4_1w_aligned = align_htf_to_ltf(prices, df_1w, l4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or
            np.isnan(h4_1w_aligned[i]) or np.isnan(l4_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
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
            # Long entry: Price breaks above H4 with volume and bullish candle
            if (close[i] > h4_1w_aligned[i] and    # Break above H4
                volume_confirmed and               # Volume spike
                bullish_candle):                   # Bullish candle
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L4 with volume and bearish candle
            elif (close[i] < l4_1w_aligned[i] and  # Break below L4
                  volume_confirmed and             # Volume spike
                  bearish_candle):                 # Bearish candle
                position = -1
                signals[i] = -0.25
    
    return signals