#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Range_Bound
Hypothesis: In the 1h timeframe, price often ranges between daily Camarilla S3/R3 levels during low volatility periods.
We buy near S3 and sell near R3 (or vice versa for shorts) only when 4h trend is aligned and volume confirms.
This mean-reversion strategy works in both bull and bear markets by fading extremes within the daily range.
Uses 4h for trend direction, 1h for entry timing, and avoids choppy markets via volume filter.
Target: 20-40 trades/year per symbol with controlled risk.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Range_Bound"
timeframe = "1h"
leverage = 1.0

def calculate_camarilla(high: float, low: float, close: float) -> Tuple[float, float, float, float]:
    """Calculate Camarilla pivot levels (R3, R4, S3, S4) from previous period's OHLC."""
    pivot = (high + low + close) / 3
    range_ = high - low
    r3 = pivot + (range_ * 1.1 / 2)
    r4 = pivot + (range_ * 1.1)
    s3 = pivot - (range_ * 1.1 / 2)
    s4 = pivot - (range_ * 1.1)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d = np.full_like(close_1d, np.nan)
    r4_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        r3, r4, s3, s4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        r3_1d[i] = r3
        r4_1d[i] = r4
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all to 1h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume filter: 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume filter: current volume > 1.2x 24-period average (avoid low-volume noise)
        volume_filter = volume[i] > 1.2 * vol_ma_24[i]
        
        # Trend filter: price above/below 4h EMA20
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Mean reversion entries: fade extremes toward S3/R3
        # Long near S3 (support) in uptrend
        long_entry = (close[i] <= s3_1d_aligned[i] * 1.005) and volume_filter and uptrend
        # Short near R3 (resistance) in downtrend
        short_entry = (close[i] >= r3_1d_aligned[i] * 0.995) and volume_filter and downtrend
        
        # Exit conditions: return to midpoint or trend failure
        mid_point = (r3_1d_aligned[i] + s3_1d_aligned[i]) / 2
        long_exit = (close[i] >= mid_point) or (not uptrend)
        short_exit = (close[i] <= mid_point) or (not downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals