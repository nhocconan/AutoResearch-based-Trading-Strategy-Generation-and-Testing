#!/usr/bin/env python3
"""
12h_1d_Pivot_R2S2_Breakout_Volume_TrendFilter
Hypothesis: Use weekly pivot point R2/S2 levels for breakout entries on 12h timeframe.
Enter long when price breaks above weekly R2 with volume and trend confirmation.
Enter short when price breaks below weekly S2 with volume and trend confirmation.
Trend filter: price above/below weekly EMA200.
Volume filter: current volume > 1.5x 20-period average.
Designed for low-frequency, high-conviction trades targeting 15-25 trades/year.
Works in bull markets by capturing breakouts and in bear markets by capturing breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points and support/resistance levels"""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    return pp, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend filter
    ema200_1w = calculate_ema(close_1w, 200)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Weekly pivot points
    pp_1w, r1_1w, r2_1w, r3_1w, s1_1w, s2_1w, s3_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Align pivot levels (no extra delay needed as pivots are based on completed weekly bar)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Uptrend: price > weekly EMA200
            if price > ema200_1w_aligned[i]:
                # Long: price breaks above weekly R2 with volume confirmation
                if price > r2_1w_aligned[i] and volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price < weekly EMA200
            elif price < ema200_1w_aligned[i]:
                # Short: price breaks below weekly S2 with volume confirmation
                if price < s2_1w_aligned[i] and volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend reversal or price returns to pivot point
            if price < ema200_1w_aligned[i] or price < pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or price returns to pivot point
            if price > ema200_1w_aligned[i] or price > pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R2S2_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0