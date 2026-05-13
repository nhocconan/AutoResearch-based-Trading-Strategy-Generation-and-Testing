# 6h_Camarilla_R4_S4_Breakout_1dTrend
# Hypothesis: Use daily Camarilla pivot points to identify breakout levels R4/S4. 
# Enter long when price breaks above R4 with volume spike and 1d EMA100 uptrend.
# Enter short when price breaks below S4 with volume spike and 1d EMA100 downtrend.
# Exit on mean reversion to daily pivot (PP). Camarilla levels are mathematically derived
# and tend to hold in ranging markets while breakouts at R4/S4 indicate strong momentum.
# Designed for low turnover (~10-25/year) on 6h timeframe to avoid fee drag.
# Works in both bull (breakouts) and bear (mean reversion to pivot) markets.

name = "6h_Camarilla_R4_S4_Breakout_1dTrend"
timeframe = "6h"
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

    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas (based on previous day's range)
    # Pivot Point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    # Resistance levels
    r1 = pp + (range_1d * 1.0833)
    r2 = pp + (range_1d * 1.1666)
    r3 = pp + (range_1d * 1.2500)
    r4 = pp + (range_1d * 1.5000)
    # Support levels
    s1 = pp - (range_1d * 1.0833)
    s2 = pp - (range_1d * 1.1666)
    s3 = pp - (range_1d * 1.2500)
    s4 = pp - (range_1d * 1.5000)
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA100 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R4 with volume spike and 1d EMA100 uptrend
            if close[i] > r4_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S4 with volume spike and 1d EMA100 downtrend
            elif close[i] < s4_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below daily pivot (mean reversion)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above daily pivot
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals