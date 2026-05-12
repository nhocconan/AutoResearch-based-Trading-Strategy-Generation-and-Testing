#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for breakout entries with 1d EMA trend filter and volume confirmation.
In bull markets, price stays above EMA and breaks above R1 for longs; in bear markets, price stays below EMA and breaks below S1 for shorts.
Camarilla levels provide precise intraday support/resistance, EMA filters trend direction, volume confirms breakout strength.
Targets 20-40 trades/year by requiring trend alignment, level break, and volume spike.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3
    range_val = high - low
    
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    s2 = close - (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    s4 = close - (range_val * 1.1 / 2)
    
    return r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla pivots and EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day
    r1, s1, r2, s2, r3, s3, r4, s4 = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    r2 = np.roll(r2, 1)
    s2 = np.roll(s2, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    r4 = np.roll(r4, 1)
    s4 = np.roll(s4, 1)
    # First day values will be invalid (rolled from last) - handled by warmup
    
    # Calculate 1d EMA for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)

    # Volume confirmation: current volume > 1.5x average of last 6 periods
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):  # Start after warmup
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine trend based on price vs EMA
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        # Breakout conditions
        breakout_above_r1 = close[i] > r1_aligned[i]
        breakout_below_s1 = close[i] < s1_aligned[i]

        if position == 0:
            # LONG: price above EMA + breakout above R1 + volume
            if price_above_ema and breakout_above_r1 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price below EMA + breakout below S1 + volume
            elif price_below_ema and breakout_below_s1 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below EMA OR breaks below S1 (reversal)
            if price_below_ema or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above EMA OR breaks above R1 (reversal)
            if price_above_ema or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals