#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + chop regime filter
# - Camarilla levels (H3, L3, H4, L4) from prior 1d: H3/L3 = mean reversion, H4/L4 = breakout
# - Long when price crosses above H3 with volume spike and chop < 61.8 (trending)
# - Short when price crosses below L3 with volume spike and chop < 61.8 (trending)
# - Exit when price reverts to opposite Camarilla level (H3/L3) or chop > 61.8 (range)
# - Uses discrete sizing (0.25) to minimize fee churn; targets 20-40 trades/year
# - Works in bull/bear: chop filter avoids whipsaws in ranging markets

name = "4h_1d_camarilla_pivot_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior day's Camarilla levels (shifted by 1 to avoid look-ahead)
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    daily_range = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * daily_range
    camarilla_h3 = close_1d + 1.1 * daily_range
    camarilla_l3 = close_1d - 1.1 * daily_range
    camarilla_l4 = close_1d - 1.5 * daily_range
    
    # Shift by 1 to use prior completed day's levels (no look-ahead)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h4[0] = np.nan
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    camarilla_l4[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h chop regime filter (EHLERS CHOPPINESS INDEX)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Chop = 100 * log10(sum(TR,14) / (max_high - min_low)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    chop = 100 * np.log10(tr_sum / (range_14 + 1e-10)) / np.log10(14)
    chop = np.where(range_14 > 0, chop, 50.0)  # default to neutral when range=0
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 (mean reversion) OR chop > 61.8 (range)
            if close_4h[i] < l3_aligned[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 (mean reversion) OR chop > 61.8 (range)
            if close_4h[i] > h3_aligned[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume and trend filter (chop < 61.8)
            if vol_spike[i] and chop[i] < 61.8:
                # Breakout long: price closes above H3
                if close_4h[i] > h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below L3
                elif close_4h[i] < l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals