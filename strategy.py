#!/usr/bin/env python3
"""
4h_ChaikinMoneyFlow_DonchianBreakout_1dTrend
Hypothesis: Chaikin Money Flow (CMF) measures institutional buying/selling pressure. 
Breakouts above Donchian(20) high/low with CMF > +0.1 (strong buying) or CMF < -0.1 (strong selling) 
capture institutional breakouts. Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts.
Volume confirmation is inherent in CMF. Designed for 20-40 trades/year on 4h timeframe to work in 
both bull (breakout longs) and bear (breakout shorts) markets.
"""

name = "4h_ChaikinMoneyFlow_DonchianBreakout_1dTrend"
timeframe = "4h"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1, hl_range)  # replace zeros with 1 to avoid div/0
    mfm = ((close - low) - (high - close)) / hl_range
    # Money Flow Volume = Money Flow Multiplier * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any value is NaN
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(cmf[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian high + strong buying pressure (CMF > 0.1) + uptrend
            if close[i] > high_max[i] and cmf[i] > 0.1 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low + strong selling pressure (CMF < -0.1) + downtrend
            elif close[i] < low_min[i] and cmf[i] < -0.1 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Breakdown below Donchian low OR loss of buying pressure (CMF < 0)
            if close[i] < low_min[i] or cmf[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Breakout above Donchian high OR loss of selling pressure (CMF > 0)
            if close[i] > high_max[i] or cmf[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals