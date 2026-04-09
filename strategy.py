#!/usr/bin/env python3
# 4h_camarilla_1d_volume_chop_v1
# Hypothesis: 4h strategy using 1d Camarilla pivot levels for mean reversion entries,
# with volume confirmation and 1d choppiness regime filter. Enters long at S3,
# short at R3, exits at S4/R4 or opposite pivot touch. Designed for low trade
# frequency (target: 20-50 total trades over 4 years) to avoid fee drag. Works in
# bull/bear by using chop filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (avoid).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_1d_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # S3 = C - (H - L) * 1.1 / 4
    # S4 = C - (H - L) * 1.1 / 2
    # R1 = C + (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    # R4 = C + (H - L) * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    s3 = close_1d - range_hl * 1.1 / 4
    s4 = close_1d - range_hl * 1.1 / 2
    r3 = close_1d + range_hl * 1.1 / 4
    r4 = close_1d + range_hl * 1.1 / 2
    
    # Align pivot levels to 4h timeframe (using previous day's levels)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Calculate choppiness index (14-period) on 1d
    # CHOP = 100 * log10(sum(ATR14) / (ATR(HH-LL))) / log10(14)
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # True range sum over 14 periods
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    hh_ll = hh - ll
    
    # Avoid division by zero
    chop = np.where(hh_ll > 0, 100 * np.log10(atr_sum / hh_ll) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S4 (take profit) or touches R3 (reverse signal)
            if close[i] <= s4_aligned[i] or close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 (take profit) or touches S3 (reverse signal)
            if close[i] >= r4_aligned[i] or close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only trade in ranging markets: CHOP > 61.8
            if chop_aligned[i] > 61.8 and volume_confirmed[i]:
                # Long at S3
                if close[i] <= s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short at R3
                elif close[i] >= r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals