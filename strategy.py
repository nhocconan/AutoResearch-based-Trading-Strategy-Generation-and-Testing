#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeChop_v2
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 or below S1 levels from prior 1d session captures institutional breakouts. Combined with 1d EMA34 trend filter, volume confirmation, and choppiness regime filter (CHOP > 61.8 = range, we avoid range). Designed for low trade frequency (~12-37/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by requiring strong trend alignment and avoiding choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels and EMA trend)
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
    
    # === Volume confirmation (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === Choppiness Index filter (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of TRUE RANGE over 14 periods
    sum_tr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sumTR14 / (hh14 - ll14)) / log10(14)
    range14 = hh14 - ll14
    # Avoid division by zero
    range14 = np.where(range14 == 0, 1e-10, range14)
    chop = 100 * np.log10(sum_tr14 / range14) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_34 = ema_34_1d_aligned[i]
        vol_spike = vol_ratio[i]
        chop_value = chop[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + above 1d EMA34 + volume > avg + NOT choppy (trending)
            if price_close > r1 and price_close > ema_34 and vol_spike > 1.0 and chop_value < 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 + below 1d EMA34 + volume > avg + NOT choppy (trending)
            elif price_close < s1 and price_close < ema_34 and vol_spike > 1.0 and chop_value < 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Time-based exit: hold max 3 bars (36h) to avoid overtrading
            # OR reverse signal
            if position == 1:
                # Exit on reverse signal or time
                if price_close < s1:  # reverse signal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > r1:  # reverse signal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeChop_v2"
timeframe = "12h"
leverage = 1.0