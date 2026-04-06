#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume + Trend Filter
Hypothesis: Camarilla pivot levels provide high-probability reversal (R3/S3) and breakout (R4/S4) levels.
Fade at R3/S3 with volume confirmation, breakout continuation at R4/S4 with trend filter.
Uses 1d Camarilla levels for context. Works in both bull and bear markets by adapting to price action.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14387_6h_camarilla_pivot_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # Using previous day's high, low, close (shifted by 1)
    ph = np.roll(high_1d, 1)
    pl = np.roll(low_1d, 1)
    pc = np.roll(close_1d, 1)
    ph[0] = high_1d[0]
    pl[0] = low_1d[0]
    pc[0] = close_1d[0]
    
    # Camarilla multipliers
    # R4 = PC + 1.1 * (PH - PL)
    # R3 = PC + 1.1 * (PH - PL) / 2
    # S3 = PC - 1.1 * (PH - PL) / 2
    # S4 = PC - 1.1 * (PH - PL)
    camarilla_range = 1.1 * (ph - pl)
    r4 = pc + camarilla_range
    r3 = pc + camarilla_range / 2
    s3 = pc - camarilla_range / 2
    s4 = pc - camarilla_range
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: require volume above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # 20% above average
    
    # Trend filter: 50-period EMA on 6h
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_up = close > ema50
    trend_down = close < ema50
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # for EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema50[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or trend reversal
            if (close[i] <= entry_price - 2.5 * atr[i] or not trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss or trend reversal
            if (close[i] >= entry_price + 2.5 * atr[i] or not trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Fade at R3/S3: price touches level with rejection
            # Breakout at R4/S4: price closes beyond level
            
            # Long setups
            long_fade = (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i] and vol_filter[i])
            long_breakout = (close[i] > r4_aligned[i] and trend_up[i] and vol_filter[i])
            
            # Short setups
            short_fade = (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i] and vol_filter[i])
            short_breakout = (close[i] < s4_aligned[i] and trend_down[i] and vol_filter[i])
            
            if long_fade or long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_fade or short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals