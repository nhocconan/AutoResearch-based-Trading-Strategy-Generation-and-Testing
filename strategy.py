#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_TrendFilter_VolumeSpike_ED
Hypothesis: 12h Camarilla pivot (R1/S1) breakout filtered by 1w EMA50 trend and volume spike (2.5x average).
Long when price > R1 and above EMA50; short when price < S1 and below EMA50.
Volume confirmation reduces false breakouts. ATR(14) stoploss (2.5x) and discrete sizing (0.25).
Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year) with low fee drag.
Uses 12h and 1w timeframes to capture multi-timeframe alignment and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 350:  # Increased warmup for 1w EMA50
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    h3_1d = df_1d_close + 1.1 * range_1d
    l3_1d = df_1d_close - 1.1 * range_1d
    h4_1d = df_1d_close + 1.382 * range_1d
    l4_1d = df_1d_close - 1.382 * range_1d
    
    # Align 1d Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # === 12h price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Volume confirmation (30-period average) ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(350, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        h3 = h3_1d_aligned[i]
        l3 = l3_1d_aligned[i]
        h4 = h4_1d_aligned[i]
        l4 = l4_1d_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.5x average (strict filter for 12h)
        volume_confirmed = volume_now > 2.5 * vol_avg
        
        if position == 0:
            # Only enter in trending markets (price > EMA50 for long, < for short)
            # Volume confirmation required to avoid false breakouts
            long_condition = (price > r1) and (price > ema_trend) and volume_confirmed
            short_condition = (price < s1) and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at H4 (extreme overbought)
            elif price > h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at L4 (extreme oversold)
            elif price < l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_TrendFilter_VolumeSpike_ED"
timeframe = "12h"
leverage = 1.0