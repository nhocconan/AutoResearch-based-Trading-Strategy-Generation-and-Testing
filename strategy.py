#!/usr/bin/env python3
"""
6h_HTF_1d_Camarilla_R3S3_Fade_v2
Hypothesis: Fade extreme intraday moves at Camarilla R3/S3 levels using 1d pivot structure.
In ranging/mean-reverting markets (common in 2025 bear), price tends to revert from R3/S3.
Use 1d trend filter (EMA34) to avoid fading strong trends. Volume confirmation ensures
fade occurs with participation. Target 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # === 1d EMA34 Trend Filter ===
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1d Camarilla Pivot Levels (from previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d_prev, 1)),
                                  np.abs(low_1d - np.roll(close_1d_prev, 1))))
    tr_1d[0] = 0  # first bar has no previous
    
    # Camarilla levels based on previous day's range
    camarilla_multiplier = 1.1 / 12  # Camarilla uses 1.1 * range / 12 for R3/S3
    camarilla_range = pd.Series(tr_1d).rolling(window=1, min_periods=1).sum().values  # daily range
    
    # Calculate R3, S3, R4, S4 from previous day's close
    r3 = close_1d_prev + camarilla_range * camarilla_multiplier * 3
    s3 = close_1d_prev - camarilla_range * camarilla_multiplier * 3
    r4 = close_1d_prev + camarilla_range * camarilla_multiplier * 4
    s4 = close_1d_prev - camarilla_range * camarilla_multiplier * 4
    
    # Align Camarilla levels to 6h timeframe (using previous day's values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) 
            or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation for fade
        
        if position == 0:
            # Fade long: price rejects at R3 (or breaks above R4 for continuation)
            if price <= r3_aligned[i] and price > s3_aligned[i] and vol_ok:
                # Additional filter: only fade if 1d trend is not strongly bullish
                if price < ema_1d_aligned[i] or abs(price - ema_1d_aligned[i]) < 0.5 * (r3_aligned[i] - s3_aligned[i]):
                    signals[i] = 0.25
                    position = 1
            # Fade short: price rejects at S3 (or breaks below S4 for continuation)
            elif price >= s3_aligned[i] and price < r3_aligned[i] and vol_ok:
                # Additional filter: only fade if 1d trend is not strongly bearish
                if price > ema_1d_aligned[i] or abs(price - ema_1d_aligned[i]) < 0.5 * (r3_aligned[i] - s3_aligned[i]):
                    signals[i] = -0.25
                    position = -1
            # Breakout continuation: strong break above R4 or below S4 with volume
            elif price > r4_aligned[i] and vol_ok and price > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif price < s4_aligned[i] and vol_ok and price < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price moves back toward mean (S3) or stops at R4
            if price >= r4_aligned[i] or price <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves back toward mean (R3) or stops at S4
            if price <= s4_aligned[i] or price >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_Camarilla_R3S3_Fade_v2"
timeframe = "6h"
leverage = 1.0