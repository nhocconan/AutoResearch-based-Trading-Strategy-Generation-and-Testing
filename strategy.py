#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d weekly pivot and volume confirmation.
- Long: TK cross above cloud + above 1d weekly pivot S3 + volume > 1.5x average
- Short: TK cross below cloud + below 1d weekly pivot R3 + volume > 1.5x average
- Exit: stop loss (2*ATR) or opposite TK cross
- Position size: 0.25
- Target: 75-200 trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14187_6h_ichimoku_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkou_a, senkou_b"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points (R1-R4, S1-S4) from daily high/low/close"""
    # Pivot Point
    pp = (high + low + close) / 3
    
    # Resistance levels
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)
    r4 = 3 * pp + (high - 3 * low)
    
    # Support levels
    s1 = 2 * pp - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)
    s4 = 3 * pp - (3 * high - low)
    
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for weekly pivot (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot on 1d data
    pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_weekly_pivot(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6h data for Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 52 for Ichimoku, 20 for volume, 14 for ATR)
    start = max(52, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or \
           np.isnan(senkou_b[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine Ichimoku cloud (green: senkou_a > senkou_b, red: senkou_a < senkou_b)
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        # TK cross: tenkan crossing kijun
        tk_cross_above = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_below = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Ichimoku signals with pivot and volume filter
        # Long: TK cross above + price above cloud + above S3 pivot + volume
        # Short: TK cross below + price below cloud + below R3 pivot + volume
        long_signal = tk_cross_above and (close[i] > cloud_top) and (close[i] > s3_aligned[i]) and vol_filter[i]
        short_signal = tk_cross_below and (close[i] < cloud_bottom) and (close[i] < r3_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or TK cross below
            if close[i] <= stop_price or tk_cross_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or TK cross above
            if close[i] >= stop_price or tk_cross_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals