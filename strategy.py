#!/usr/bin/env python3
"""
12h_1D_Camarilla_Pivot_R3S3_Breakout_Trend_Volume
Hypothesis: On 12h timeframe, trade breakouts of Camarilla R3/S3 levels derived from the previous 1d candle,
only in the direction of the 1d EMA34 trend, with volume confirmation (>1.5x 10-period average).
This structure has shown strong performance in ETH/SOL historically. Targets 15-30 trades/year.
Works in bull/bear by following 1d trend, avoids whipsaws via trend filter and volume confirmation.
"""

name = "12h_1D_Camarilla_Pivot_R3S3_Breakout_Trend_Volume"
timeframe = "12h"
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

    # Get 1d data for Camarilla levels and EMA34 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Calculate 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_34_prev_aligned = align_htf_to_ltf(prices, df_1d, ema_34_prev)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # 12h volume spike: current > 1.5x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA warmup
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_prev_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > Camarilla R3 + uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and
                ema_34_aligned[i] > ema_34_prev_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price < Camarilla S3 + downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and
                  ema_34_aligned[i] < ema_34_prev_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Camarilla S3 (mean reversion) or trend change
            if (close[i] < camarilla_s3_aligned[i] or
                ema_34_aligned[i] < ema_34_prev_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Camarilla R3 (mean reversion) or trend change
            if (close[i] > camarilla_r3_aligned[i] or
                ema_34_aligned[i] > ema_34_prev_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals