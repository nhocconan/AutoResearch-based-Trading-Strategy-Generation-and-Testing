#!/usr/bin/env python3
"""
6h_PivotPoint_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v1
Hypothesis: 6h breakout at daily pivot R1/S1 levels filtered by 1d EMA50 trend and volume spike (>1.8x 24-period average).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn.
1d trend filter provides robust directional bias across bull/bear markets while reducing whipsaws.
Target: 12-30 trades/year per symbol for low fee drag and strong test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter and pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d OHLC for pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate daily pivot and R1/S1 levels (standard formula)
    # Pivot = (high + low + close) / 3
    # R1 = 2*Pivot - low
    # S1 = 2*Pivot - high
    pivot_1d = (df_1d_high + df_1d_low + df_1d_close) / 3.0
    r1_1d = 2 * pivot_1d - df_1d_low
    s1_1d = 2 * pivot_1d - df_1d_high
    
    # Align 1d pivot levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter: 24-period average (4 days of 6h bars) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        if position == 0:
            # Volume filter: current volume > 1.8x 24-period average
            vol_filter = vol_current > 1.8 * vol_average
            
            # Long conditions: price > R1 (breakout), 1d uptrend, volume filter
            long_breakout = price > r1_1d_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < S1 (breakdown), 1d downtrend, volume filter
            short_breakout = price < s1_1d_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic - ONLY enter on volume filter + trend alignment
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
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
            # Check stoploss
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

name = "6h_PivotPoint_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0