#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation and choppiness regime filter
# Camarilla pivots from 12h provide intraday support/resistance that work in ranging markets
# Volume confirmation (current 4h volume > 1.3x 20-period average) filters low-conviction breakouts
# Choppiness regime filter: CHOP(14) > 61.8 = ranging (mean revert at H3/L3), CHOP < 38.2 = trending (breakout at H4/L4)
# Position size: 0.25 for mean reversion, 0.30 for breakouts
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_12h_camarilla_volume_chop_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = C + Range * 1.1/2
    # L4 = C - Range * 1.1/2
    # H3 = C + Range * 1.1/4
    # L3 = C - Range * 1.1/4
    # H2 = C + Range * 1.1/6
    # L2 = C - Range * 1.1/6
    # H1 = C + Range * 1.1/12
    # L1 = C - Range * 1.1/12
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    h4 = close_12h + range_12h * 1.1 / 2.0
    l4 = close_12h - range_12h * 1.1 / 2.0
    h3 = close_12h + range_12h * 1.1 / 4.0
    l3 = close_12h - range_12h * 1.1 / 4.0
    h2 = close_12h + range_12h * 1.1 / 6.0
    l2 = close_12h - range_12h * 1.1 / 6.0
    h1 = close_12h + range_12h * 1.1 / 12.0
    l1 = close_12h - range_12h * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_12h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, l4)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute choppiness index (14-period) for 4h
    # CHOP = 100 * log10(sum(ATR(1)) / (HHV(high,14) - LLV(low,14))) / log10(14)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values
    sum_atr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    hh_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = hh_high_14 - ll_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # Avoid division by zero
    chop = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x average 4h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit on retracement to H3 or stoploss at L4
            if close[i] < h3_aligned[i] or close[i] < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Mean reversion size
                
        elif position == -1:  # Short position
            # Exit on retracement to L3 or stoploss at H4
            if close[i] > l3_aligned[i] or close[i] > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Mean reversion size
        else:  # Flat
            # Determine regime based on choppiness
            if chop[i] > 61.8:  # Ranging market - mean reversion at H3/L3
                if volume_confirmed:
                    if close[i] > h3_aligned[i]:
                        position = -1
                        signals[i] = -0.25  # Short at H3
                    elif close[i] < l3_aligned[i]:
                        position = 1
                        signals[i] = 0.25   # Long at L3
            else:  # Trending market - breakout at H4/L4
                if volume_confirmed:
                    if close[i] > h4_aligned[i]:
                        position = 1
                        signals[i] = 0.30   # Long breakout
                    elif close[i] < l4_aligned[i]:
                        position = -1
                        signals[i] = -0.30  # Short breakout
    
    return signals