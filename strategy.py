#!/usr/bin/env python3
"""
6h_Pivot_Bounce_Momentum
Hypothesis: Price bouncing off daily pivot levels (R1/S1) with 6h momentum (MACD) and volume confirmation captures mean reversion in ranging markets and continuation in trending markets. Works in both bull and bear by using pivot as dynamic S/R and momentum as filter.
Target: 15-35 trades/year per symbol.
"""

name = "6h_Pivot_Bounce_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily pivot levels from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivot points for each day using previous day's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = 2*Pivot - L
    # S1 = 2*Pivot - H
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 6h momentum: MACD histogram (12,26,9)
    close_series = pd.Series(close)
    ema12 = close_series.ewm(span=12, adjust=False, min_periods=12).values
    ema26 = close_series.ewm(span=26, adjust=False, min_periods=26).values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).values
    macd_hist = macd_line - signal_line
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        px = close[i]
        piv = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        macd = macd_hist[i]
        vol_ok = volume_conf[i]
        
        if position == 0:
            # LONG: price near S1 with bullish momentum and volume
            if px >= s1_val * 0.995 and px <= s1_val * 1.005 and macd > 0 and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: price near R1 with bearish momentum and volume
            elif px <= r1_val * 1.005 and px >= r1_val * 0.995 and macd < 0 and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches pivot or momentum turns bearish
            if px >= piv * 0.995 or macd < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches pivot or momentum turns bullish
            if px <= piv * 1.005 or macd > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals