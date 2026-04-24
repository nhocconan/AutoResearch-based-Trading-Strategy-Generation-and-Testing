#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d Camarilla pivot levels (H3, L3) for breakout signals.
- Volume: Current 12h volume > 1.5 * 20-period 12h volume MA to confirm breakout strength.
- Regime: 12h Choppiness Index (CHOP) > 61.8 for ranging market (mean reversion at H3/L3),
          CHOP < 38.2 for trending market (breakout continuation).
- Entry: Long when price breaks above H3 AND volume spike AND CHOP < 61.8 (not extreme chop).
         Short when price breaks below L3 AND volume spike AND CHOP < 61.8.
- Exit: Opposite breakout (price < L3 for long, price > H3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Camarilla pivots work well in crypto due to institutional respect for these levels,
and volume/regime filters reduce false breakouts in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    prev_close = np.roll(df_1d_close, 1)
    prev_high = np.roll(df_1d_high, 1)
    prev_low = np.roll(df_1d_low, 1)
    # First day has no previous data
    prev_close[0] = df_1d_close[0]
    prev_high[0] = df_1d_high[0]
    prev_low[0] = df_1d_low[0]
    
    range_hl = prev_high - prev_low
    H3 = prev_close + 1.1 * range_hl / 4
    L3 = prev_close - 1.1 * range_hl / 4
    
    # Calculate 12h Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR over period) / (max_high - min_low over period)) / log10(period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close_12h = np.roll(close, 1)
    prev_close_12h[0] = close[0]
    tr = true_range(high, low, prev_close_12h)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    for i in range(14, n):
        if max_high_14[i] != min_low_14[i]:
            chop[i] = 100 * np.log10(np.sum(atr_14[i-13:i+1]) / (max_high_14[i] - min_low_14[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when range is zero
    chop[:14] = 50  # not enough data
    
    # Calculate 20-period volume MA on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need volume MA and CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period volume MA
        volume_spike = volume[i] > (1.5 * vol_ma)
        
        # Regime filter: avoid extreme chop (CHOP > 61.8) where breakouts fail
        not_extreme_chop = chop_val < 61.8
        
        if position == 0:
            # Check for entry signals with volume spike and regime filter
            if volume_spike and not_extreme_chop:
                # Bullish: price breaks above H3
                if curr_low > H3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below L3
                elif curr_high < L3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR loss of volume confirmation OR extreme chop
            if curr_high < L3_aligned[i] or not volume_spike or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR loss of volume confirmation OR extreme chop
            if curr_low > H3_aligned[i] or not volume_spike or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0