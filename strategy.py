#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + 1w choppiness regime
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels from prior 1d with volume confirmation
# - Volume filter: 1d volume > 1.5x 20-period average volume (ensures institutional participation)
# - Regime filter: 1w Choppiness Index < 50 (trending market favors breakout continuation)
# - In trending markets (CHOP < 50): breakout continuation in direction of break
# - In ranging markets (CHOP >= 50): fade Camarilla extremes toward H4/L4 (mean reversion)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(20)

name = "12h_1d_1w_camarilla_vol_chop_v1"
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
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
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
    
    # Pre-compute 1d Camarilla pivot levels (based on prior day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low)*1.1/2
    # H3 = close + 1.25*(high-low)*1.1/2
    # L3 = close - 1.25*(high-low)*1.1/2
    # L4 = close - 1.5*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.25 * range_1d * 1.1 / 2
    camarilla_l3 = close_1d - 1.25 * range_1d * 1.1 / 2
    camarilla_h4 = close_1d + 1.5 * range_1d * 1.1 / 2
    camarilla_l4 = close_1d - 1.5 * range_1d * 1.1 / 2
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 12h ATR(20) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_20 = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Camarilla mean reversion OR stoploss hit
            if close_12h[i] < camarilla_h4_aligned[i] or close_12h[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Camarilla mean reversion OR stoploss hit
            if close_12h[i] > camarilla_l4_aligned[i] or close_12h[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume filter and chop regime
            # In trending markets (CHOP < 50): breakout continuation
            # In ranging markets (CHOP >= 50): mean reversion at Camarilla extremes
            if vol_spike_aligned[i]:
                if chop_aligned[i] < 50.0:  # trending market - breakout continuation
                    # Long: price breaks above Camarilla H3
                    if close_12h[i] > camarilla_h3_aligned[i]:
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: price breaks below Camarilla L3
                    elif close_12h[i] < camarilla_l3_aligned[i]:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
                else:  # ranging market - mean reversion
                    # Long: price at lower Camarilla L3
                    if close_12h[i] <= camarilla_l3_aligned[i] * 1.0005:  # tiny buffer for noise
                        position = 1
                        entry_price = close_12h[i]
                        signals[i] = 0.25
                    # Short: price at upper Camarilla H3
                    elif close_12h[i] >= camarilla_h3_aligned[i] * 0.9995:
                        position = -1
                        entry_price = close_12h[i]
                        signals[i] = -0.25
    
    return signals