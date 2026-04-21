#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: 4h Camarilla pivot (R1/S1) breakout filtered by 1d EMA50 trend and choppiness regime (CHOP > 61.8 = range, < 38.2 = trend).
In ranging markets (CHOP > 61.8): mean reversion at S1/R1 (long at S1, short at R1).
In trending markets (CHOP < 38.2): breakout continuation (long above R1, short below S1).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to balance returns and fee drag.
Designed to work in both bull and bear markets via adaptive regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend and CHOP)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Choppiness Index (CHOP) ===
    atr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            atr_1d.append(0.0)
        else:
            tr = max(
                df_1d_high[i] - df_1d_low[i],
                abs(df_1d_high[i] - df_1d_close[i-1]),
                abs(df_1d_low[i] - df_1d_close[i-1])
            )
            atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_14 - lowest_14
    chop_1d = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) 
            or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        chop = chop_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Determine market regime
            is_ranging = chop > 61.8
            is_trending = chop < 38.2
            
            # Ranging market: mean reversion at S1/R1
            if is_ranging:
                long_condition = price <= s1 * 1.002  # slight buffer for entry
                short_condition = price >= r1 * 0.998
            
            # Trending market: breakout continuation
            elif is_trending:
                long_condition = price > r1 and price > ema_trend
                short_condition = price < s1 and price < ema_trend
            
            # Choppy transition zone: no entries
            else:
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime-based exit
            elif (chop > 61.8 and price >= r1 * 0.998) or (chop < 38.2 and price < s1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime-based exit
            elif (chop > 61.8 and price <= s1 * 1.002) or (chop < 38.2 and price > r1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0