#!/usr/bin/env python3
"""
6h_LiquiditySweep_Retest_1dTrend
Hypothesis: In BTC/ETH, price often sweeps liquidity (equal highs/lows) before reversing. 
Buy when price sweeps prior day's low then closes back above it with 1d uptrend and volume confirmation.
Sell when price sweeps prior day's high then closes back below it with 1d downtrend and volume confirmation.
Uses 1d trend filter to align with higher timeframe momentum. Targets 15-25 trades/year.
"""

name = "6h_LiquiditySweep_Retest_1dTrend"
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
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Previous day's high and low (for liquidity levels)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)

    # Align liquidity levels to 6h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 6h bar
        ema50 = ema50_1d_aligned[i]
        ph = prev_high_aligned[i]
        pl = prev_low_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(ph) or 
            np.isnan(pl) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Sweep prior day's low then close back above it
            # Condition: low swept below pl AND close recovered above pl
            if (low[i] < pl and close[i] > pl and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Sweep prior day's high then close back below it
            elif (high[i] > ph and close[i] < ph and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below prior day's low or trend fails
            if (close[i] < pl or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above prior day's high or trend fails
            if (close[i] > ph or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals