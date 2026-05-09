#!/usr/bin/env python3
# 1d_WickReversal_1wHighLow_Volume
# Hypothesis: On 1d timeframe, trade reversals at weekly high/low wicks.
# Long when price closes below weekly low but reverses back above it with volume.
# Short when price closes above weekly high but reverses back below it with volume.
# Weekly structure provides strong support/resistance; intraday rejection signals exhaustion.
# Works in both bull (buy dips at weekly support) and bear (sell rallies at weekly resistance).
# Target: 20-60 trades over 4 years (5-15/year) with size 0.25.

name = "1d_WickReversal_1wHighLow_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly high and low from completed weekly bars
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align to daily timeframe (use previous week's completed high/low)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Wick rejection signals:
    # Long setup: price pierces weekly low but closes back above it (bullish rejection)
    pierce_low = low < weekly_low_aligned
    close_above_low = close > weekly_low_aligned
    long_setup = pierce_low & close_above_low
    
    # Short setup: price pierces weekly high but closes back below it (bearish rejection)
    pierce_high = high > weekly_high_aligned
    close_below_high = close < weekly_high_aligned
    short_setup = pierce_high & close_below_high
    
    # Volume confirmation: current volume > 1.5x 20-day average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume average
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(long_setup[i]) or np.isnan(short_setup[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish rejection at weekly low + volume
            if long_setup[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish rejection at weekly high + volume
            elif short_setup[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish rejection at weekly high or loss of momentum
            if short_setup[i] or (close[i] < weekly_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish rejection at weekly low or loss of momentum
            if long_setup[i] or (close[i] > weekly_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals