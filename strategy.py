#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_chop_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels (L3, H3) with volume confirmation and choppiness regime filter.
# Long when price crosses above H3 with volume > 1.5x 20-period average and CHOP > 61.8 (range).
# Short when price crosses below L3 with volume confirmation and CHOP > 61.8.
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 12-30 trades/year.
# Uses 1d HTF data for Camarilla levels, called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 day for pivot calculation
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation (use shift to avoid look-ahead)
    prev_high = np.roll(high_d, 1)
    prev_low = np.roll(low_d, 1)
    prev_close = np.roll(close_d, 1)
    prev_high[0] = np.nan  # First day has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot levels
    # H3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    # L3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    hl_range = prev_high - prev_low
    h3 = prev_close + 1.1 * hl_range / 4.0
    l3 = prev_close - 1.1 * hl_range / 4.0
    
    # Align daily Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR1) / (n * max(high-low))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_1 = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    atr_1[0] = high[0] - low[0]  # First bar
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high_low = pd.Series(high - low).rolling(window=14, min_periods=14).max().values
    chop = 100 * np.log10(atr_sum / (14 * max_high_low + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: CHOP > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below L3 OR chop regime ends (trending)
            if close[i] < l3_aligned[i] or chop[i] <= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above H3 OR chop regime ends (trending)
            if close[i] > h3_aligned[i] or chop[i] <= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_filter:
                # Long entry: price crosses above H3
                if close[i] > h3_aligned[i] and close[i-1] <= h3_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price crosses below L3
                elif close[i] < l3_aligned[i] and close[i-1] >= l3_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals