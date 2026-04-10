#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume confirmation + choppiness regime filter
# - Primary signal: Price touches Camarilla H3 (resistance) for short, L3 (support) for long
# - Volume filter: 12h volume > 1.5x 20-period average (avoid low-volume false breakouts)
# - Regime filter: Choppiness Index(14) on 1d < 38.2 (trending market) for breakout trades
# - Position size: 0.25 discrete level
# - Stoploss: 2.0x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla levels adapt to volatility; chop filter avoids ranging markets

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
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3 = pivot + (range_1d * 1.1 / 4.0)  # Resistance level 3
    l3 = pivot - (range_1d * 1.1 / 4.0)  # Support level 3
    h4 = pivot + (range_1d * 1.1 / 2.0)  # Resistance level 4
    l4 = pivot - (range_1d * 1.1 / 2.0)  # Support level 4
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute 12h volume spike filter
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * avg_volume_20)
    
    # Pre-compute 1d Choppiness Index for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(n) * (HHV - LLV)) / log10(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 13 and sum_atr_14[i] > 0 and hh_14[i] > ll_14[i]:
            chop[i] = 100 * np.log10(sum_atr_14[i] / np.log10(14) * (hh_14[i] - ll_14[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # Neutral value when insufficient data
    
    # Trending market: Chop < 38.2
    chop_trending = chop < 38.2
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # Pre-compute 12h ATR(14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(chop_trending_aligned[i]) or
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below L3 OR stoploss hit
            if close_12h[i] < l3_aligned[i] or close_12h[i] < entry_price - 2.0 * atr_14_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above H3 OR stoploss hit
            if close_12h[i] > h3_aligned[i] or close_12h[i] > entry_price + 2.0 * atr_14_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touch with volume and regime filters
            if vol_spike[i] and chop_trending_aligned[i]:
                # Long: Price touches L3 support from above
                if close_12h[i] <= l3_aligned[i] * 1.001 and close_12h[i-1] > l3_aligned[i-1]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: Price touches H3 resistance from below
                elif close_12h[i] >= h3_aligned[i] * 0.999 and close_12h[i-1] < h3_aligned[i-1]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals