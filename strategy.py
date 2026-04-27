#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeChop
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter, volume confirmation, and chop regime filter.
Designed for 4h timeframe targeting 75-200 total trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn.
In trending regimes (price > EMA50 for longs, < EMA50 for shorts) and low chop (CHOP < 50),
breakouts at R1/S1 with volume spike capture strong momentum continuations.
Exit on trend reversal (close crosses EMA50) or chop expansion (CHOP > 61.8).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d data for HTF filters
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous 1d bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index on 4h: CHOP > 61.8 = range, CHOP < 38.2 = trend
    def choppiness_index(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first TR
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        chop = np.where((max_high - min_low) > 0, 
                        100 * np.log10(atr * window / (max_high - min_low)) / np.log10(window), 
                        50)
        return chop
    
    chop = choppiness_index(high, low, close, window=14)
    chop_filter = chop < 50  # Only trade in low/moderate chop (avoid extreme ranging)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 1d shift, 12h EMA50, vol avg, chop
    start_idx = max(30, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        in_low_chop = chop_filter[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with EMA alignment, volume spike, and chop filter
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            in_low_chop)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             in_low_chop)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below EMA50 (trend reversal) OR chop expands (range bound)
            if close_val < ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA50 (trend reversal) OR chop expands (range bound)
            if close_val > ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeChop"
timeframe = "4h"
leverage = 1.0