# 12h_1D_Camarilla_R3S3_Breakout_With_Trend_Filter
# Hypothesis: Camarilla pivot levels from daily chart provide strong support/resistance.
# Breakout above R3 or below S3 with volume confirmation and aligned with 1d trend (price above/below EMA34)
# captures institutional breakouts while avoiding false signals in ranging markets.
# Trend filter ensures we only trade in direction of higher timeframe momentum.
# Designed for low frequency (target 12-37 trades/year) to minimize fee drag on 12h timeframe.

name = "12h_1D_Camarilla_R3S3_Breakout_With_Trend_Filter"
timeframe = "12h"
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

    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Camarilla formulas
    range_ = prev_high - prev_low
    R3 = prev_close + (range_ * 1.1 / 2)
    S3 = prev_close - (range_ * 1.1 / 2)

    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume spike: volume > 2.0 * 24-period average (~12 days at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + price breaks above R3 + volume spike
            if close[i] > ema34_1d_aligned[i] and close[i] > R3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price breaks below S3 + volume spike
            elif close[i] < ema34_1d_aligned[i] and close[i] < S3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R3 or trend turns bearish
            if close[i] < R3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S3 or trend turns bullish
            if close[i] > S3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3