#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout + 1w volume spike + 1w chop regime filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 1d
# - Volume confirmation: 1w volume > 2.0x 20-period average volume (high-participation breakouts only)
# - Regime filter: 1w Choppiness Index > 61.8 (range) enables fade of H3/L3 touches; < 38.2 (trend) enables breakout continuation
# - Works in bull/bear: In trending markets, breakouts persist; in ranging markets, fade false breakouts at pivot levels
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 30-100 trades over 4 years (7-25/year) per 1d strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20) on 1d

name = "1d_1w_camarilla_volume_chop_v1"
timeframe = "1d"
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
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    # We use H3/L3 for breakouts, H4/L4 for stronger breaks
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 1d timeframe (already aligned, but shift by 1 for previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4, additional_delay_bars=1)
    
    # Pre-compute 1w volume spike filter
    volume_1w = df_1w['volume'].values
    avg_volume_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1w > (2.0 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    
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
    
    # Pre-compute 1d ATR(20) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_20 = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: mean reversion at Camarilla levels OR stoploss hit
            if (close_1d[i] < camarilla_h3_aligned[i] or 
                close_1d[i] < entry_price - 2.0 * atr_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: mean reversion at Camarilla levels OR stoploss hit
            if (close_1d[i] > camarilla_l3_aligned[i] or 
                close_1d[i] > entry_price + 2.0 * atr_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume spike and chop regime filter
            # In ranging markets (CHOP > 61.8): fade Camarilla touches (mean reversion)
            # In trending markets (CHOP < 38.2): breakout continuation
            if volume_spike_aligned[i]:
                if chop_aligned[i] > 61.8:  # ranging market - mean reversion
                    # Long: price touches lower Camarilla L3 level
                    if close_1d[i] <= camarilla_l3_aligned[i] * 1.001:  # small buffer for noise
                        position = 1
                        entry_price = close_1d[i]
                        signals[i] = 0.25
                    # Short: price touches upper Camarilla H3 level
                    elif close_1d[i] >= camarilla_h3_aligned[i] * 0.999:
                        position = -1
                        entry_price = close_1d[i]
                        signals[i] = -0.25
                elif chop_aligned[i] < 38.2:  # trending market - breakout continuation
                    # Long: price breaks above upper Camarilla H4 level
                    if close_1d[i] > camarilla_h4_aligned[i]:
                        position = 1
                        entry_price = close_1d[i]
                        signals[i] = 0.25
                    # Short: price breaks below lower Camarilla L4 level
                    elif close_1d[i] < camarilla_l4_aligned[i]:
                        position = -1
                        entry_price = close_1d[i]
                        signals[i] = -0.25
    
    return signals