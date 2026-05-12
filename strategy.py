#!/usr/bin/env python3
# 4h_ParabolicSAR_Trend_Filter_Volume
# Hypothesis: Parabolic SAR captures trend direction and potential reversals. 
# Long when price > SAR and SAR is rising, short when price < SAR and SAR is falling.
# Filtered by 1d EMA50 trend direction for multi-timeframe alignment.
# Volume confirmation ensures institutional participation. 
# Works in bull markets via longs in uptrends, bear markets via shorts in downtrends.
# Target: 20-30 trades/year.

name = "4h_ParabolicSAR_Trend_Filter_Volume"
timeframe = "4h"
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
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Parabolic SAR calculation (0.02 step, 0.2 max)
    sar = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = 0  # extreme point
    sar[0] = low[0]

    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if sar[i] > low[i]:
                trend[i] = -1
                sar[i] = ep
                ep = high[i]
                af = 0.02
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
                if sar[i] > low[i]:
                    sar[i] = low[i]
        else:  # downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if sar[i] < high[i]:
                trend[i] = 1
                sar[i] = ep
                ep = low[i]
                af = 0.02
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
                if sar[i] < high[i]:
                    sar[i] = high[i]

    # Volume spike: current > 2.0x average of last 6 bars (1 day)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(sar[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > SAR (uptrend) + 1d EMA50 uptrend + volume spike
            if (close[i] > sar[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price < SAR (downtrend) + 1d EMA50 downtrend + volume spike
            elif (close[i] < sar[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < SAR (trend reversal)
            if close[i] < sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > SAR (trend reversal)
            if close[i] > sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals