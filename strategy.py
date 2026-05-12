#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) on 1d act as strong support/resistance. Breakouts above R3 or below S3 with 1w trend filter (EMA13) and volume confirmation capture explosive moves in both bull and bear markets. Uses 1d timeframe with 1w EMA13 trend filter for higher timeframe context. Volume > 1.5x 20-period average confirms breakout strength.
"""

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Using previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First value: use current day's values (no look-ahead)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + range_val * 1.1 / 2.0  # R3 = pivot + (high-low)*1.1/2
    s3 = pivot - range_val * 1.1 / 2.0  # S3 = pivot - (high-low)*1.1/2

    # 1w EMA13 trend filter
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after volume MA warmup
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_13_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + 1w uptrend + volume confirmation
            if (close[i] > r3[i] and 
                close[i] > ema_13_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 1w downtrend + volume confirmation
            elif (close[i] < s3[i] and 
                  close[i] < ema_13_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 (re-test of support) or trend reversal
            if close[i] < s3[i] or close[i] < ema_13_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 (re-test of resistance) or trend reversal
            if close[i] > r3[i] or close[i] > ema_13_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals