#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeChop_v2
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 or below S1 levels from prior 1d session captures institutional breakouts. Combined with 1d EMA34 trend filter and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend). Designed for low trade frequency (~12-30/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels, EMA trend, and chop filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1-day EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla levels from prior 1-day session (HLC of previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Choppiness Index (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr / (max_high - min_low)) / log10(14)
    range_14 = max_high - min_low
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(sum_tr / range_14) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_34 = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + above 1d EMA34 + trending regime (CHOP < 38.2)
            if price_close > r1 and price_close > ema_34 and chop_val < 38.2:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 + below 1d EMA34 + trending regime (CHOP < 38.2)
            elif price_close < s1 and price_close < ema_34 and chop_val < 38.2:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit: reverse signal or chop becomes too high (choppy/ranging market)
            if position == 1:
                if price_close < s1 or chop_val > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > r1 or chop_val > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeChop_v2"
timeframe = "12h"
leverage = 1.0