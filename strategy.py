# 6h_Camarilla_R4_S4_Breakout_1dTrend_Volume
# Hypothesis: Price reacts strongly at extended Camarilla levels (R4/S4) derived from 1d timeframe.
# R4 = C + (H-L) * 1.1/2, S4 = C - (H-L) * 1.1/2 represent breakout levels.
# Go long when price breaks above R4 with 1d uptrend (close > EMA34) and volume confirmation.
# Go short when price breaks below S4 with 1d downtrend (close < EMA34) and volume confirmation.
# Uses 1d timeframe for structure and 6h for execution, targeting 12-37 trades/year.
# Works in bull markets (breakouts above R4 in uptrend) and bear markets (breakdowns below S4 in downtrend).
# Volume spike filters false breakouts, EMA34 ensures trend alignment.

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla pivot calculation and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R4, S4) from previous 1d bar
    # R4 = C + (H-L) * 1.1/2
    # S4 = C - (H-L) * 1.1/2
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_width = (high_1d - low_1d) * 1.1 / 2
    r4 = close_1d + camarilla_width
    s4 = close_1d - camarilla_width
    
    # 1d trend: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 2.0 * 6-period average (1 day worth at 6h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R4 + 1d uptrend + volume spike
            if close[i] > r4_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S4 + 1d downtrend + volume spike
            elif close[i] < s4_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S4 or trend reversal
            if close[i] < s4_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R4 or trend reversal
            if close[i] > r4_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3