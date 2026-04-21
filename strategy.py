#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v2
Hypothesis: 4h Camarilla pivot (R1/S1) breakout filtered by 1d EMA50 trend and choppiness regime (CHOP < 61.8 = trending). Only take breakout trades in trending markets aligned with 1d EMA50 direction. Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25). Designed for BTC/ETH to work in both bull and bear markets by requiring trend alignment. Timeframe: 4h, uses 1d HTF for trend, Camarilla, and regime filter.
Target: 75-200 total trades over 4 years = 19-50/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend, Camarilla, and choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === 1d OHLC for indicators ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Camarilla pivot levels (based on previous 1d bar) ===
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    # True Range
    tr1 = pd.Series(df_1d_high - df_1d_low)
    tr2 = pd.Series(np.abs(df_1d_high - np.roll(df_1d_close, 1)))
    tr3 = pd.Series(np.abs(df_1d_low - np.roll(df_1d_close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).sum().values  # Sum of TR over 14 periods
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(atr_1d / (hh_1d - ll_1d)) / log10(14)
    # Avoid division by zero
    hl_range = hh_1d - ll_1d
    chop_1d = np.where(hl_range > 0, 100 * np.log10(atr_1d / hl_range) / np.log10(14), 100)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === ATR (14-period) for 4h timeframe stoploss ===
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        chop = chop_1d_aligned[i]
        
        # Regime filter: only trade when market is trending (CHOP < 61.8)
        is_trending = chop < 61.8
        
        if position == 0:
            # Only enter in direction of 1d trend with regime filter
            long_condition = (price > r1) and (price > ema_trend) and is_trending
            short_condition = (price < s1) and (price < ema_trend) and is_trending
            
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
            # Trend reversal or regime change exit
            elif price < ema_trend or chop >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal or regime change exit
            elif price > ema_trend or chop >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v2"
timeframe = "4h"
leverage = 1.0