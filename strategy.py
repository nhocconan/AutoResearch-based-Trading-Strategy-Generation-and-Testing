# 12h_ParabolicSAR_Trend_Following_With_Volume_Filter
# Hypothesis: Parabolic SAR identifies trend direction and reversals with clear entry/exit points.
# Combined with volume filter to avoid false signals and ensure institutional participation.
# Works in bull markets by capturing uptrends and in bear markets by capturing downtrends.
# Uses 1d timeframe for trend filter and 12h for execution to maintain low trade frequency.

name = "12h_ParabolicSAR_Trend_Following_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate 50-period EMA on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Parabolic SAR calculation on 12h data
    # Initialize
    psar = np.zeros(n)
    psar[0] = low[0]
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = high[0] if trend == 1 else low[0]  # extreme point

    for i in range(1, n):
        if trend == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't exceed previous two lows
            if i >= 2:
                psar[i] = min(psar[i], low[i-1], low[i-2])
            # Reverse if price breaks below SAR
            if low[i] < psar[i]:
                trend = -1
                psar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                # Update EP and AF
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't go below previous two highs
            if i >= 2:
                psar[i] = max(psar[i], high[i-1], high[i-2])
            # Reverse if price breaks above SAR
            if high[i] > psar[i]:
                trend = 1
                psar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                # Update EP and AF
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)

    # Volume filter: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(psar[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend (price above 1d EMA50) + price above PSAR + volume filter
            if close[i] > ema50_1d_aligned[i] and close[i] > psar[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price below 1d EMA50) + price below PSAR + volume filter
            elif close[i] < ema50_1d_aligned[i] and close[i] < psar[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below PSAR or trend turns bearish
            if close[i] < psar[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above PSAR or trend turns bullish
            if close[i] > psar[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3