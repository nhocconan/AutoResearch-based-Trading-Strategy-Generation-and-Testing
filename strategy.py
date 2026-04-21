#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_v2
Hypothesis: 4h Camarilla pivot (R1/S1) breakout filtered by 1d EMA50 trend and chop regime (EHLERS chop > 61.8 = range, < 38.2 = trend).
Adds chop filter to avoid whipsaws in ranging markets while maintaining trend-following edge.
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to reduce fee drag.
Designed for both bull and bear markets via 1d trend filter and volatility-adjusted exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend and chop filter)
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
    
    # === EHLERS CHOPPINESS INDEX (14-period) on 1d ===
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # Where ATR(1) = true range, ATR(n) = n-period ATR
    high_1d = df_1d_high
    low_1d = df_1d_low
    close_1d = df_1d_close
    
    # True range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # first bar
    
    # Sum of ATR(1) over 14 periods
    sum_tr1 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # ATR(14) on 1d
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Chop formula: 100 * log10(sum_tr1 / (14 * atr_14_1d)) / log10(14)
    chop_1d = 100 * np.log10(sum_tr1 / (14 * atr_14_1d + 1e-10)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Chop regime: > 61.8 = ranging (avoid entries), < 38.2 = trending (allow entries)
    chop_ranging = chop_1d_aligned > 61.8
    chop_trending = chop_1d_aligned < 38.2
    
    # === ATR (14-period) for stoploss on 4h ===
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
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Entry conditions: breakout + trend filter + chop regime (trending only)
            long_breakout = price > r1_1d_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            long_chop = chop_trending[i]  # only allow in trending regime
            
            short_breakout = price < s1_1d_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            short_chop = chop_trending[i]  # only allow in trending regime
            
            # Entry logic
            if long_breakout and long_trend and long_chop:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and short_chop:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below S1 (breakdown)
            elif price < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout)
            elif price > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0