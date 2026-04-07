#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + Choppiness Regime
Long/short at Camarilla pivot levels (S3/S4, R3/R4) with volume confirmation
Only trade in trending regimes (Choppiness Index < 38.2)
Exit when price reaches opposite pivot level or closes back inside S2/R2
Timeframe: 12h, HTF: 1d for Choppiness filter
Target: 50-150 total trades over 4 years (12-37/year)
Works in bull/bear: pivot levels act as support/resistance, volume confirms breakouts,
choppiness filter avoids ranging markets where pivots fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Previous day's OHLC for Camarilla (from 1d data) ===
    df_1d = get_htf_data(prices, '1d')
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # === Camarilla Pivot Levels (based on previous day) ===
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S4 = C - ((H - L) * 1.5)
    # S3 = C - ((H - L) * 1.25)
    # S2 = C - ((H - L) * 1.166)
    # S1 = C - ((H - L) * 1.083)
    # R1 = C + ((H - L) * 1.083)
    # R2 = C + ((H - L) * 1.166)
    # R3 = C + ((H - L) * 1.25)
    # R4 = C + ((H - L) * 1.5)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    
    s4 = prev_close - (rng * 1.5)
    s3 = prev_close - (rng * 1.25)
    s2 = prev_close - (rng * 1.166)
    s1 = prev_close - (rng * 1.083)
    r1 = prev_close + (rng * 1.083)
    r2 = prev_close + (rng * 1.166)
    r3 = prev_close + (rng * 1.25)
    r4 = prev_close + (rng * 1.5)
    
    # === Align Camarilla levels to 12h timeframe (shifted by 1 for completed day) ===
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Choppiness Index (14-period) from 1d data for regime filter ===
    # CHOP = 100 * log10(sum(ATR) / (max(HH) - min(LL))) / log10(period)
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - np.roll(prev_close, 1))
    tr3 = np.abs(prev_low - np.roll(prev_close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    hh = pd.Series(prev_high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(prev_low).rolling(window=14, min_periods=14).min()
    chop_raw = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    chop = chop_raw.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when NOT choppy (trending market)
        # Chop < 38.2 = trending, Chop > 61.8 = ranging
        if chop_aligned[i] > 38.2:
            # In ranging or transition zone, stay flat
            if position == 1:
                # Exit long if price closes below S2
                if close[i] < s2_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price closes above R2
                if close[i] > r2_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Chop <= 38.2: Trending regime - look for pivot breakouts
        
        if position == 1:  # Long position
            # Exit: price reaches R4 (take profit) or closes back below S2 (stop)
            if close[i] >= r4_aligned[i] or close[i] < s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S4 (take profit) or closes back above R2 (stop)
            if close[i] <= s4_aligned[i] or close[i] > r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Breakout of S3 or R3 with volume
            # Short at S3/S4 breakdown, Long at R3/R4 breakout
            if close[i] <= s3_aligned[i] and close[i] > s4_aligned[i]:
                # Breakdown below S3 with volume -> short
                position = -1
                signals[i] = -0.25
            elif close[i] >= r3_aligned[i] and close[i] < r4_aligned[i]:
                # Breakout above R3 with volume -> long
                position = 1
                signals[i] = 0.25
    
    return signals