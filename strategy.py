#!/usr/bin/env python3
# 4h_camarilla_1d_volume_chop_v5
# Hypothesis: 4h strategy using Camarilla pivot levels from 1d HTF for entry/exit,
# with volume confirmation and 1d choppiness regime filter. Long when price touches
# Camarilla S3 in trending market (CHOP<38.2) with volume spike; short when touches
# S3 in ranging market (CHOP>61.8) for mean reversion. Uses discrete sizing (±0.25)
# to limit trade frequency (~30-60/year) and minimize fee drag. Works in bull/bear
# by adapting to regime: trend following in trending markets, mean reversion in ranging.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_1d_volume_chop_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    # L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    # We focus on L3 (support) and H3 (resistance) for mean reversion/breakout
    prev_high_1d = pd.Series(high_1d).shift(1)
    prev_low_1d = pd.Series(low_1d).shift(1)
    prev_close_1d = pd.Series(close_1d).shift(1)
    
    # Avoid look-ahead: use previous day's values
    prev_high_1d = prev_high_1d.values
    prev_low_1d = prev_low_1d.values
    prev_close_1d = prev_close_1d.values
    
    # Calculate pivot levels
    prev_range = prev_high_1d - prev_low_1d
    camarilla_h3 = prev_close_1d + 1.125 * prev_range
    camarilla_l3 = prev_close_1d - 1.125 * prev_range
    camarilla_h4 = prev_close_1d + 1.5 * prev_range
    camarilla_l4 = prev_close_1d - 1.5 * prev_range
    
    # Align HTF levels to LTF (4h)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) = EMA of TR
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (profit target) or breaks below L4 (stop)
            if close[i] >= h3_aligned[i] or close[i] <= l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 (profit target) or breaks above H4 (stop)
            if close[i] <= l3_aligned[i] or close[i] >= h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed[i]:
                # Regime-based logic using Choppiness Index
                # CHOP < 38.2 = strong trend (trend follow)
                # CHOP > 61.8 = ranging market (mean revert)
                if chop_aligned[i] < 38.2:  # Trending market
                    # Long: price touches L3 with volume spike (breakout/retest)
                    if close[i] <= l3_aligned[i] * 1.001:  # Allow small buffer
                        position = 1
                        signals[i] = 0.25
                    # Short: price touches H3 with volume spike (breakdown/retest)
                    elif close[i] >= h3_aligned[i] * 0.999:
                        position = -1
                        signals[i] = -0.25
                elif chop_aligned[i] > 61.8:  # Ranging market
                    # Mean reversion: long at L3, short at H3
                    if close[i] <= l3_aligned[i] * 1.001:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] >= h3_aligned[i] * 0.999:
                        position = -1
                        signals[i] = -0.25
    
    return signals