#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_chop_v1
# Hypothesis: 12h strategy using Camarilla pivot levels from daily timeframe for structure,
# with volume confirmation (>1.5x 20-period average) and chop regime filter (CHOP(14) > 61.8 for ranging).
# Long at L3 support with volume in choppy market; short at H3 resistance with volume in choppy market.
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 12-37 trades/year.
# Uses 1d HTF data for Camarilla levels and chop filter, called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_chop_v1"
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
    
    # 1d HTF data for Camarilla pivot and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for daily calculations
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Daily Camarilla pivot levels (based on previous day's OHLC)
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # L3 = close - (range * 1.1/4)
    # L4 = close - (range * 1.1/2)
    # H3 = close + (range * 1.1/4)
    # H4 = close + (range * 1.1/2)
    # Note: We use previous day's data to avoid look-ahead
    prev_high = np.roll(high_d, 1)
    prev_low = np.roll(low_d, 1)
    prev_close = np.roll(close_d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    
    # Camarilla levels
    l3 = prev_close - (rng * 1.1 / 4)
    l4 = prev_close - (rng * 1.1 / 2)
    h3 = prev_close + (rng * 1.1 / 4)
    h4 = prev_close + (rng * 1.1 / 2)
    
    # Daily Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    # We'll use a simplified version: high-low range based
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    atr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    max_high = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    # Chop calculation: avoid division by zero
    chop = np.zeros_like(close_d)
    mask = range_hl > 0
    chop[mask] = 100 * np.log10(sum_atr[mask] / range_hl[mask]) / np.log10(14)
    chop[~mask] = 50  # Neutral when range is zero
    
    # Align daily data to 12h timeframe
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion at pivots)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price rises above H3 (take profit) OR falls below L4 (stop loss)
            if close[i] > h3_aligned[i] or close[i] < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price falls below L3 (take profit) OR rises above H4 (stop loss)
            if close[i] < l3_aligned[i] or close[i] > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_filter:
                # Long entry: price at L3 support with volume in choppy market
                if abs(close[i] - l3_aligned[i]) < (h3_aligned[i] - l3_aligned[i]) * 0.02:  # Within 2% of L3
                    position = 1
                    signals[i] = 0.25
                # Short entry: price at H3 resistance with volume in choppy market
                elif abs(close[i] - h3_aligned[i]) < (h3_aligned[i] - l3_aligned[i]) * 0.02:  # Within 2% of H3
                    position = -1
                    signals[i] = -0.25
    
    return signals