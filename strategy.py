#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels from 1w HTF + volume confirmation + choppiness regime filter
# - Uses 1w Camarilla pivot levels (H3, L3, H4, L4) as key support/resistance
# - Long when price touches L3/L4 with volume spike and chop > 61.8 (range market)
# - Short when price touches H3/H4 with volume spike and chop > 61.8 (range market)
# - Exit when price moves to opposite H3/L3 level or chop < 38.2 (trending market)
# - Designed for 1d timeframe: targets 10-30 trades/year to avoid fee drag
# - Works in ranging markets (chop > 61.8) where pivot reversals are strongest
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "1d_1w_camarilla_pivot_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1w Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Calculate ranges
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    h4 = pp + (range_1w * 1.1 / 2)
    h3 = pp + (range_1w * 1.1 / 4)
    l3 = pp - (range_1w * 1.1 / 4)
    l4 = pp - (range_1w * 1.1 / 2)
    
    # Align HTF Camarilla levels to LTF
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Pre-compute 1d volume confirmation
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * avg_volume_20)
    
    # Pre-compute 1d choppiness index (CHOP) for regime filter
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14) for CHOP denominator
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # Using rolling window of 14 periods
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # Avoid division by zero
    chop = np.where(range_14 > 0, 100 * np.log10(atr_sum / range_14) / np.log10(14), 50)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Regime filters: CHOP > 61.8 = ranging (good for mean reversion), CHOP < 38.2 = trending
    chop_ranging = chop > 61.8
    chop_trending = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(l4_aligned[i]) or np.isnan(vol_spike[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (opposite level) OR market starts trending
            if close_1d[i] >= h3_aligned[i] or chop_trending[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 (opposite level) OR market starts trending
            if close_1d[i] <= l3_aligned[i] or chop_trending[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries in ranging market
            if chop_ranging[i] and vol_spike[i]:
                # Long when price touches L4/L3 with volume spike
                if close_1d[i] <= l4_aligned[i] or close_1d[i] <= l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short when price touches H4/H3 with volume spike
                elif close_1d[i] >= h4_aligned[i] or close_1d[i] >= h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals