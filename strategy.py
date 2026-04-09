#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume spike + choppiness regime filter
# Long when price touches Camarilla L3 support with volume spike in choppy market (mean reversion)
# Short when price touches Camarilla H3 resistance with volume spike in choppy market
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion in chop, volume confirms genuine interest

name = "4h_1d_camarilla_pivot_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6, L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12, L1 = close - 1.1*(high-low)/12
    # Pivot = (high + low + close)/3
    
    rng = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_h3 = camarilla_pivot + 1.1 * rng / 4.0
    camarilla_l3 = camarilla_pivot - 1.1 * rng / 4.0
    
    # Calculate 1d average volume (20-period) for volume spike detection
    vol_1d = df_1d['volume'].values
    vol_s_1d = pd.Series(vol_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate choppiness index on 4h (14-period) for regime filter
    def true_range(h, l, pc):
        return np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # Sum of true range over 14 periods
    tr_s = pd.Series(tr)
    tr_sum_14 = tr_s.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market
    denominator = hh_14 - ll_14
    chop = np.full(n, np.nan)
    mask = (denominator > 0) & ~np.isnan(tr_sum_14) & ~np.isnan(denominator)
    chop[mask] = 100 * np.log10(tr_sum_14[mask] / denominator[mask]) / np.log10(14)
    
    # Volume confirmation: current 4h volume > 2.0x average 4h volume (20-period)
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy market (CHOP > 61.8)
        if chop[i] <= 61.8:
            # Exit position if market becomes trending
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price moves above L3 (mean reversion complete) or stops volume spike
            if close[i] > camarilla_l3_aligned[i] or not volume_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price moves below H3 (mean reversion complete) or stops volume spike
            if close[i] < camarilla_h3_aligned[i] or not volume_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion entry: price at L3/H3 with volume spike in choppy market
            if abs(close[i] - camarilla_l3_aligned[i]) < 0.001 * close[i] and volume_spike[i]:
                # Price touching L3 support -> long
                position = 1
                signals[i] = 0.25
            elif abs(close[i] - camarilla_h3_aligned[i]) < 0.001 * close[i] and volume_spike[i]:
                # Price touching H3 resistance -> short
                position = -1
                signals[i] = -0.25
    
    return signals