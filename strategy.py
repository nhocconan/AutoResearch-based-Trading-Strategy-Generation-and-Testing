#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_Breakout_Volume_V1
Hypothesis: 6h strategy using weekly pivot points (from 1w HTF) for key support/resistance levels,
combined with 1d EMA50 trend filter and volume confirmation (>1.5x 20-period average).
Enter long when price breaks above weekly R1 with 1d uptrend and volume spike.
Enter short when price breaks below weekly S1 with 1d downtrend and volume spike.
Exit on opposite weekly level break or ATR(14) trailing stop (2.0*ATR).
Target: 12-25 trades/year (~50-100 total over 4 years) to minimize fee drag.
Works in bull/bear via HTF pivot structure and 1d trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivots, 1d for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 5 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1w Weekly Pivot Points (R1, S1) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Standard pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align to 6h timeframe (use previous completed weekly bar)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) 
            or np.isnan(volume_threshold[i]) or np.isnan(atr[i]) 
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Long conditions: price > weekly R1, 1d uptrend, volume spike
            long_breakout = price > r1_1w_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            long_volume = volume_6h[i] > volume_threshold[i]
            
            # Short conditions: price < weekly S1, 1d downtrend, volume spike
            short_breakout = price < s1_1w_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            short_volume = volume_6h[i] > volume_threshold[i]
            
            # Entry logic
            if long_breakout and long_trend and long_volume:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and short_volume:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below weekly S1 (support broken)
            elif price < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above weekly R1 (resistance broken)
            elif price > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DailyTrend_Breakout_Volume_V1"
timeframe = "6h"
leverage = 1.0