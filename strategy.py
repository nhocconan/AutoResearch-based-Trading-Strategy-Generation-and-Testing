#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter
Hypothesis: Camarilla R1/S1 breakout on 1d with 1w EMA50 trend filter and choppiness regime filter.
Designed for 1d timeframe targeting 30-100 total trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear markets:
- Trending regime (CHOP < 38.2): trade breakouts in direction of 1w EMA50
- Ranging regime (CHOP >= 38.2): avoid false breakouts
Exit on close crossing 1w EMA50.
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Camarilla levels from previous 1d bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 1d
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1w EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR) / (HHV - LLV)) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (hh - ll + 1e-10)) / np.log10(14)  # simplified for per-bar
    # Correct per-bar CHOP calculation
    sum_atr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_atr / (hh - ll + 1e-10)) / np.log10(14))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 1d shift, 1w EMA50, CHOP
    start_idx = max(30, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        chop_val = chop[i]
        
        # Regime filter: only trade breakouts in trending market (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with EMA alignment and regime filter
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            is_trending)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             is_trending)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below 1w EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1w EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter"
timeframe = "1d"
leverage = 1.0