#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS
# Hypothesis: Use 1-day Camarilla pivot levels with S1 and R1 breakouts as entry signals.
# Filter trades using 1-day EMA34 trend direction and volume confirmation.
# Designed to capture high-probability breakout trades in both bull and bear markets
# while minimizing whipsaws through trend and volume filters.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for a given period."""
    range_val = high - low
    if range_val == 0:
        return np.array([close, close, close, close, close, close, close, close])
    c = close
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    r2 = c + range_val * 1.1 / 6
    r1 = c + range_val * 1.1 / 12
    s1 = c - range_val * 1.1 / 12
    s2 = c - range_val * 1.1 / 6
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return np.array([s4, s3, s2, s1, c, r1, r2, r3, r4])

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    camarilla_levels = np.array([
        calculate_camarilla(h, l, c) 
        for h, l, c in zip(df_1d['high'], df_1d['low'], df_1d['close'])
    ])
    
    # Extract S1 (index 3) and R1 (index 5) levels
    s1_levels = camarilla_levels[:, 3]  # S1 level
    r1_levels = camarilla_levels[:, 5]  # R1 level
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_ratio = volume / (vol_ma + 1e-9)  # Avoid division by zero

    # Align 1d indicators to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_levels)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA warmup
        # Skip if any required value is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        bullish_breakout = close[i] > r1_aligned[i]  # Price breaks above R1
        bearish_breakout = close[i] < s1_aligned[i]  # Price breaks below S1
        
        # Trend filter: price relative to EMA34
        price_above_ema34 = close[i] > ema34_aligned[i]
        price_below_ema34 = close[i] < ema34_aligned[i]
        
        # Volume confirmation: significant volume spike
        volume_confirm = volume_ratio[i] > 1.5

        if position == 0:
            # LONG: Bullish breakout + price above EMA34 + volume confirmation
            if bullish_breakout and price_above_ema34 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish breakout + price below EMA34 + volume confirmation
            elif bearish_breakout and price_below_ema34 and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR loses EMA34 support
            if close[i] < s1_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR loses EMA34 resistance
            if close[i] > r1_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals