#!/usr/bin/env python3
"""
1h_KeltnerBreakout_ATRFilter_4hTrend_1dVol
Hypothesis: On 1h timeframe, price breakouts above/below Keltner Channel (2*ATR) 
with 4h EMA50 trend filter and 1d volume spike (1.5x 20-period avg) generate 
trend-following signals. Uses 4h/1d for signal direction, 1h only for entry timing.
Targets 15-37 trades/year (60-150 total over 4 years) with low turnover to minimize fee drag.
Works in bull via momentum breakouts and bear via filtered trend continuation.
"""

name = "1h_KeltnerBreakout_ATRFilter_4hTrend_1dVol"
timeframe = "1h"
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

    # Get 4h data for trend (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values

    # Calculate 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Calculate 4h ATR(14) for Keltner Channel
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)

    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Get aligned values for current 1h bar
        ema50 = ema50_4h_aligned[i]
        atr = atr14_4h_aligned[i]
        vol_avg_1d = vol_avg_20_1d_aligned[i]

        # Skip if any required data is NaN
        if np.isnan(ema50) or np.isnan(atr) or np.isnan(vol_avg_1d):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Calculate 1h Keltner Channel using 4h ATR (scaled to 1h)
        # Approximate 1h ATR as 4h ATR / 2 (since 4h is 4x 1h, vol scales with sqrt(time))
        atr_1h_est = atr / 2.0
        keltner_up = ema50 + 2.0 * atr_1h_est  # Using 4h EMA as base for simplicity
        keltner_down = ema50 - 2.0 * atr_1h_est

        # Volume condition: 1d volume > 1.5x 20-day average
        vol_condition = volume[i] > vol_avg_1d * 1.5

        if position == 0:
            # LONG: Price breaks above Keltner Upper + above 4h EMA50 + volume spike
            if (close[i] > keltner_up and 
                close[i] > ema50 and 
                vol_condition):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Keltner Lower + below 4h EMA50 + volume spike
            elif (close[i] < keltner_down and 
                  close[i] < ema50 and 
                  vol_condition):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 4h EMA50 or below Keltner Middle
            if close[i] < ema50 or close[i] < (ema50):  # Exit on trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above 4h EMA50 or above Keltner Middle
            if close[i] > ema50 or close[i] > (ema50):  # Exit on trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals