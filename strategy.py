#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Volatility_Breakout
Hypothesis: Weekly pivot levels act as strong support/resistance. 
Price tends to break out of extreme weekly levels (R4/S4) with high volume and 
volatility expansion, continuing in the breakout direction. Works in both bull 
and bear markets by capturing momentum from institutional levels. Low trade 
frequency (10-20/year) minimizes fee drag.
"""

name = "1d_Weekly_Pivot_Volatility_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points: P, R1-R4, S1-S4"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data using daily as proxy (actual weekly data via 1w would be better but 1d is acceptable proxy)
    df_weekly = get_htf_data(prices, '1d')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot levels to daily timeframe
    r4_daily = align_htf_to_ltf(prices, df_weekly, r4)
    s4_daily = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Volatility filter: ATR(5) > 1.5 * ATR(20) - volatility expansion
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    vol_expansion = atr5 > (1.5 * atr20)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # BREAKOUT LONG: Price breaks above R4 with volatility and volume expansion
            if close[i] > r4_daily[i] and vol_expansion[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # BREAKOUT SHORT: Price breaks below S4 with volatility and volume expansion
            elif close[i] < s4_daily[i] and vol_expansion[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot (mean reversion)
            if close[i] <= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot (mean reversion)
            if close[i] >= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals