#!/usr/bin/env python3
# 6h_ParabolicSAR_Trend_Filter_12h
# Hypothesis: Parabolic SAR on 6h with 12h EMA trend filter. Go long when SAR flips below price and 12h EMA is rising; short when SAR flips above price and 12h EMA is falling. This captures trend reversals with trend alignment to reduce whipsaws. Works in bull markets (catching uptrends) and bear markets (catching downtrends). Target: 15-35 trades/year per symbol.

name = "6h_ParabolicSAR_Trend_Filter_12h"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Parabolic SAR parameters
    af_start = 0.02
    af_step = 0.02
    af_max = 0.2

    # Initialize SAR arrays
    sar = np.full(n, np.nan)
    trend = np.full(n, 0)  # 1 for uptrend, -1 for downtrend
    af = np.full(n, af_start)
    ep = np.full(n, 0.0)   # extreme point

    # Set initial values
    if high[1] > high[0]:
        trend[0] = 1
        sar[0] = low[0]
        ep[0] = high[1]
    else:
        trend[0] = -1
        sar[0] = high[0]
        ep[0] = low[1]

    # Calculate SAR
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # SAR cannot be above the lowest low of the past two periods
            sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            if low[i] < sar[i]:  # trend reversal
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # SAR cannot be below the highest high of the past two periods
            sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
            if high[i] > sar[i]:  # trend reversal
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema12_12h = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_12h_prev = np.roll(ema12_12h, 1)
    ema12_12h_prev[0] = ema12_12h[0]
    ema12_rising = ema12_12h > ema12_12h_prev
    ema12_falling = ema12_12h < ema12_12h_prev

    # Align 12h indicators to 6h timeframe
    ema12_rising_aligned = align_htf_to_ltf(prices, df_12h, ema12_rising)
    ema12_falling_aligned = align_htf_to_ltf(prices, df_12h, ema12_falling)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if SAR or trend is not yet valid
        if np.isnan(sar[i]) or trend[i] == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: SAR flips below price (bullish) + 12h EMA rising
            if trend[i] == 1 and sar[i] < close[i] and ema12_rising_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: SAR flips above price (bearish) + 12h EMA falling
            elif trend[i] == -1 and sar[i] > close[i] and ema12_falling_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: SAR flips above price (bearish reversal)
            if trend[i] == -1 and sar[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: SAR flips below price (bullish reversal)
            if trend[i] == 1 and sar[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals