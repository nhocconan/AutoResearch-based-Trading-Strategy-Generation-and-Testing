#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_ChopFilter_v4
Hypothesis: On 12h timeframe, trade Camarilla R1/S1 breakouts only when aligned with 1d trend and low chop regime (range-bound market). Uses discrete sizing (0.25) to minimize fee drag. Designed for 12-37 trades/year to avoid overtrading. Works in bull/bear via trend filter - only long in uptrend, short in downtrend. Chop filter avoids whipsaws in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, EMA34 trend, and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d chop filter (Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sum_tr_14 / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d_prev = df_1d['high'].values
    low_1d_prev = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla width
    rang = high_1d_prev - low_1d_prev
    
    # Resistance levels R1, R2, R3, R4
    r1 = close_1d_prev + rang * 1.1 / 12
    r2 = close_1d_prev + rang * 1.1 / 6
    r3 = close_1d_prev + rang * 1.1 / 4
    r4 = close_1d_prev + rang * 1.1 / 2
    
    # Support levels S1, S2, S3, S4
    s1 = close_1d_prev - rang * 1.1 / 12
    s2 = close_1d_prev - rang * 1.1 / 6
    s3 = close_1d_prev - rang * 1.1 / 4
    s4 = close_1d_prev - rang * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detector (20-bar volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Chop filter: only trade when chop < 61.8 (trending) OR chop > 61.8 (ranging) - we use both regimes
        # But avoid extreme chop > 80 (too choppy) and chop < 30 (too strong trend)
        chop_ok = (chop_aligned[i] >= 30) & (chop_aligned[i] <= 80)
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in uptrend AND chop OK
            if close[i] > r1_aligned[i] and volume_spike[i] and uptrend and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in downtrend AND chop OK
            elif close[i] < s1_aligned[i] and volume_spike[i] and downtrend and chop_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R1 OR trend changes OR chop too high
            if close[i] < r1_aligned[i] or not uptrend or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S1 OR trend changes OR chop too high
            if close[i] > s1_aligned[i] or not downtrend or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_ChopFilter_v4"
timeframe = "12h"
leverage = 1.0