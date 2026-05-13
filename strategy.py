#!/usr/bin/env python3
# 1D_Camarilla_Pivot_MeanReversion_1wTrend
# Hypothesis: Price reverting to daily Camarilla Pivot (S1/R1) with weekly trend filter and volume confirmation works in both bull and bear markets. 
# Weekly trend ensures we trade with the higher timeframe momentum, while mean reversion at pivot levels provides high-probability entries with tight stops.

name = "1D_Camarilla_Pivot_MeanReversion_1wTrend"
timeframe = "1d"
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

    # 1w EMA50 for trend filter (load once, align)
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Calculate Camarilla levels for current day (using previous day's range)
        if i > 0:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            if range_val > 0:
                camarilla_multiplier = 1.1 / 12
                r1 = prev_close + range_val * camarilla_multiplier * 1
                s1 = prev_close - range_val * camarilla_multiplier * 1
                r3 = prev_close + range_val * camarilla_multiplier * 4
                s3 = prev_close - range_val * camarilla_multiplier * 4
            else:
                r1 = s1 = r3 = s3 = prev_close
        else:
            r1 = s1 = r3 = s3 = close[0]

        if position == 0:
            # LONG: Price near S1 (support) + weekly uptrend + volume confirmation
            if (close[i] <= s1 * 1.005 and  # Within 0.5% of S1
                close[i] > ema50_1w_aligned[i] and  # Weekly uptrend
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near R1 (resistance) + weekly downtrend + volume confirmation
            elif (close[i] >= r1 * 0.995 and  # Within 0.5% of R1
                  close[i] < ema50_1w_aligned[i] and  # Weekly downtrend
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Pivot or R3 or trend changes
            if (close[i] >= (r1 + s1) / 2 or  # Reached midpoint (pivot area)
                close[i] >= r3 or             # Hit strong resistance
                close[i] < ema50_1w_aligned[i]):  # Weekly trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Pivot or S3 or trend changes
            if (close[i] <= (r1 + s1) / 2 or  # Reached midpoint (pivot area)
                close[i] <= s3 or             # Hit strong support
                close[i] > ema50_1w_aligned[i]):  # Weekly trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals