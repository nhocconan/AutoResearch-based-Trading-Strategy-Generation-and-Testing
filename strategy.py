#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + 1w chop regime filter
# - Primary signal: Williams Alligator (Jaw/Teeth/Lips) alignment on 12h
#   * Long: Lips > Teeth > Jaw (bullish alignment)
#   * Short: Lips < Teeth < Jaw (bearish alignment)
# - Volume confirmation: 1d volume > 1.8x 20-period average volume (strong participation)
# - Regime filter: 1w Choppiness Index > 61.8 (range market) enables mean reversion at extremes
#   * In ranging markets (CHOP > 61.8): trade Alligator alignment with price near extremes
#   * In trending markets (CHOP < 38.2): trade strong Alligator alignment
# - Works in bull/bear: Alligator catches trends early; chop filter avoids whipsaws in ranges
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "12h_1d_1w_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1w Choppiness Index
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    chop_raw = np.where((hh_14 - ll_14) > 0,
                        100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14),
                        50)  # neutral when no range
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw, additional_delay_bars=0)
    
    # Pre-compute 12h Williams Alligator
    # Jaw: Smoothed Median Price (13, 8)
    # Teeth: Smoothed Median Price (8, 5)
    # Lips: Smoothed Median Price (5, 3)
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Jaw (13, 8)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).median().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (8, 5)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).median().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips (5, 3)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).median().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Alligator alignment signals
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    bullish_alignment = lips_above_teeth & teeth_above_jaw
    bearish_alignment = lips_below_teeth & teeth_below_jaw
    
    bullish_aligned = align_htf_to_ltf(prices, prices, bullish_alignment.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, prices, bearish_alignment.astype(float))
    
    # Pre-compute 12h ATR(14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator reversal OR stoploss hit
            if (not bullish_aligned[i]) or close_12h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator reversal OR stoploss hit
            if (not bearish_aligned[i]) or close_12h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with volume spike and chop regime filter
            # In ranging markets (CHOP > 61.8): trade alignment with price near extremes
            # In trending markets (CHOP < 38.2): trade strong alignment
            if volume_spike_aligned[i]:
                if chop_aligned[i] > 61.8:  # ranging market
                    # Long: bullish alignment + price near low (mean reversion long)
                    if bullish_aligned[i] and close_12h[i] <= low_12h[i] * 1.002:
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: bearish alignment + price near high (mean reversion short)
                    elif bearish_aligned[i] and close_12h[i] >= high_12h[i] * 0.998:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
                elif chop_aligned[i] < 38.2:  # trending market
                    # Long: strong bullish alignment
                    if bullish_aligned[i]:
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: strong bearish alignment
                    elif bearish_aligned[i]:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
    
    return signals