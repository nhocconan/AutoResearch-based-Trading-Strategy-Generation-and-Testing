#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + volume confirmation + 1d choppiness regime filter
# Uses 12h price action at 1-day Camarilla pivot levels (H3/L3) for institutional breakout trades,
# volume to confirm institutional participation, and 1d Choppiness Index to avoid ranging markets.
# Targets 50-150 total trades over 4 years (12-37/year) with disciplined entries.
# Works in both bull and bear by only taking breakouts in direction of 1d trend (EMA50).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for Camarilla pivots and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Resistance levels: H3 = C + (H-L)*1.1/2, H4 = C + (H-L)*1.1
    # Support levels: L3 = C - (H-L)*1.1/2, L4 = C - (H-L)*1.1
    h3_1d = close_1d + (range_1d * 1.1 / 2)
    l3_1d = close_1d - (range_1d * 1.1 / 2)
    h4_1d = close_1d + (range_1d * 1.1)
    l4_1d = close_1d - (range_1d * 1.1)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Choppiness Index (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above H3 (strong resistance) + volume spike + chop < 61.8 (trending) + price above EMA50
        if (high[i] > h3_1d_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            chop_aligned[i] < 61.8 and
            close[i] > ema50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below L3 (strong support) + volume spike + chop < 61.8 (trending) + price below EMA50
        elif (low[i] < l3_1d_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              chop_aligned[i] < 61.8 and
              close[i] < ema50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or chop > 61.8 (ranging market)
        elif position == 1 and (low[i] < l3_1d_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (high[i] > h3_1d_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Volume_Chop_Filter"
timeframe = "12h"
leverage = 1.0