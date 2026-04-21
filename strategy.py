#!/usr/bin/env python3
"""
1d_KAMA_Trend_WeeklyPivot_Filter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
filtered by weekly Camarilla pivot levels (R1/S1) for mean-reversion entries.
Long when price > KAMA and touches weekly S1; short when price < KAMA and touches weekly R1.
Volume confirmation (1.5x average) reduces false signals. ATR-based stoploss (2.0x) manages risk.
Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull/bear markets.
Timeframe: 1d, uses 1w HTF for pivot calculation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for Camarilla pivots and trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w OHLC for Camarilla pivot calculation (based on previous weekly bar) ===
    df_1w_open = df_1w['open'].values
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each weekly bar
    range_1w = df_1w_high - df_1w_low
    r1_1w = df_1w_close + 0.275 * range_1w
    s1_1w = df_1w_close - 0.275 * range_1w
    h3_1w = df_1w_close + 1.1 * range_1w
    l3_1w = df_1w_close - 1.1 * range_1w
    h4_1w = df_1w_close + 1.382 * range_1w
    l4_1w = df_1w_close - 1.382 * range_1w
    
    # Align 1w Camarilla levels to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    h4_1w_aligned = align_htf_to_ltf(prices, df_1w, h4_1w)
    l4_1w_aligned = align_htf_to_ltf(prices, df_1w, l4_1w)
    
    # === Daily KAMA for trend direction ===
    close = prices['close'].values
    # Efficiency ratio over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        kama_val = kama[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        h3 = h3_1w_aligned[i]
        l3 = l3_1w_aligned[i]
        h4 = h4_1w_aligned[i]
        l4 = l4_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Mean reversion entries at weekly S1/R1 with KAMA trend filter
            long_condition = (price <= s1 * 1.005) and (price > kama_val) and volume_confirmed
            short_condition = (price >= r1 * 0.995) and (price < kama_val) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < kama_val:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at weekly H3 (overbought)
            elif price >= h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > kama_val:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at weekly L3 (oversold)
            elif price <= l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_WeeklyPivot_Filter_v1"
timeframe = "1d"
leverage = 1.0