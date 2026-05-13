#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Trend_Volume
Hypothesis: Breakout above weekly pivot resistance or below weekly pivot support
with daily trend alignment and volume confirmation works across market regimes.
Weekly pivots provide structural support/resistance; breakouts capture momentum.
Trend filter avoids counter-trend entries; volume confirms institutional interest.
Target: 15-25 trades per year per symbol.
"""

name = "1d_WeeklyPivot_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points from weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Pivot point: (H + L + C) / 3
    wp = (wh + wl + wc) / 3.0
    # Resistance 1: (2 * P) - L
    r1 = 2 * wp - wl
    # Support 1: (2 * P) - H
    s1 = 2 * wp - wh
    
    # Align weekly levels to daily timeframe
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        up_trend = uptrend[i]
        down_trend = downtrend[i]
        vol_ok = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1 with uptrend and volume
            if close[i] > r1_val and up_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with downtrend and volume
            elif close[i] < s1_val and down_trend and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below pivot or trend reverses
            if close[i] < wp_aligned[i] or not up_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above pivot or trend reverses
            if close[i] > wp_aligned[i] or not down_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals