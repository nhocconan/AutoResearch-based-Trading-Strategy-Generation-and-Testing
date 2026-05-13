#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Trend_Volume_Signal
Hypothesis: Camarilla pivot levels (R1/S1) on daily chart act as key support/resistance.
Breakouts above R1 or below S1 with volume confirmation and trend filter (daily EMA50) capture
trend continuation moves. Works in bull markets via breakouts and bear markets via reversals at
extremes, with low trade frequency to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_Trend_Volume_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r1 = close + (range_ * 1.1 / 12)
    s1 = close - (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    s2 = close - (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)
    s3 = close - (range_ * 1.1 / 4)
    r4 = close + (range_ * 1.1 / 2)
    s4 = close - (range_ * 1.1 / 2)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_daily = get_htf_data(prices, '1d')
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla levels from previous day's data
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
        daily_high, daily_low, daily_close
    )
    
    # Align Camarilla levels to 4h timeframe (using previous day's close for alignment)
    r1_4h = align_htf_to_ltf(prices, df_daily, r1)
    s1_4h = align_htf_to_ltf(prices, df_daily, s1)
    
    # Daily EMA50 for trend filter
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        if position == 0:
            # LONG: Break above R1 with volume and above EMA50
            if close[i] > r1_4h[i] and close[i] > ema50_4h[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume and below EMA50
            elif close[i] < s1_4h[i] and close[i] < ema50_4h[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S1 or reaches R2 (take profit)
            if close[i] < s1_4h[i] or close[i] >= r2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R1 or reaches S2 (take profit)
            if close[i] > r1_4h[i] or close[i] <= s2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Calculate R2 and S2 for exit conditions
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
        daily_high, daily_low, daily_close
    )
    r2_4h = align_htf_to_ltf(prices, df_daily, r2)
    s2_4h = align_htf_to_ltf(prices, df_daily, s2)