#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + chop regime filter
# - Primary signal: price touches Camarilla H3 (short) or L3 (long) from prior 1d
# - Volume confirmation: 12h volume > 1.5x 20-period median volume (avoid low-participation)
# - Regime filter: 1d choppiness index > 61.8 (range market) for mean reversion at pivots
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla pivots work in ranging markets; chop filter ensures
#   we only mean revert in ranging conditions, avoiding strong trends where pivots fail

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Camarilla pivots and chop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior 1d Camarilla levels (H3, L3, H4, L4)
    # Camarilla: Range = high - low
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    # H4 = close + (high - low) * 1.1/2
    # L4 = close - (high - low) * 1.1/2
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    camarilla_h4 = close_1d + range_1d * 1.1 / 2
    camarilla_l4 = close_1d - range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d Choppiness Index (CHOP) - measures if market is choppy (range) or trending
    # CHOP > 61.8 = ranging (good for mean reversion at pivots)
    # CHOP < 38.2 = trending (avoid mean reversion)
    atr_1d = pd.Series(
        np.maximum(
            np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    ).rolling(window=14, min_periods=14).sum().values
    
    sum_true_range_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(
        (highest_high_14 - lowest_low_14) == 0,
        50.0,  # neutral when no range
        100 * np.log10(sum_true_range_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume regime: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (median_volume_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches L4 (profit target) or crosses above H3 (stop/reversal)
            if low[i] <= l4_aligned[i] or high[i] >= h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches H4 (profit target) or crosses below L3 (stop/reversal)
            if high[i] >= h4_aligned[i] or low[i] <= l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla touch with volume confirmation and chop regime
            # Long: price touches L3 (or below) AND volume regime AND chop > 61.8 (ranging)
            if low[i] <= l3_aligned[i] and volume_regime[i] and chop_aligned[i] > 61.8:
                position = 1
                signals[i] = 0.25
            # Short: price touches H3 (or above) AND volume regime AND chop > 61.8 (ranging)
            elif high[i] >= h3_aligned[i] and volume_regime[i] and chop_aligned[i] > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals