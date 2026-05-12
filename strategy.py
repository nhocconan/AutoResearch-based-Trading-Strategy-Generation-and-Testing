#!/usr/bin/env python3
"""
1d_KeltnerChannel_WeeklyTrend_Volume
Hypothesis: Trade breakouts above upper Keltner Channel or below lower Keltner Channel on 1d timeframe when aligned with weekly EMA40 trend and confirmed by volume spike. Uses ATR-based channel for volatility adaptation, works in both bull and bear markets by following weekly trend. Targets 10-25 trades/year with strict confluence of breakout, trend alignment, and volume confirmation.
Timeframe: 1d
"""

name = "1d_KeltnerChannel_WeeklyTrend_Volume"
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

    # Get weekly data for EMA40 trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    # Calculate weekly EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)

    # Calculate ATR(20) for Keltner Channel on daily data
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # Keltner Channel: EMA20 ± 2*ATR
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr

    # Volume spike: current > 2.0x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):  # Start after EMA40 and ATR20 warmup
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > upper KC + price > weekly EMA40 + volume spike
            if (close[i] > kc_upper[i] and 
                close[i] > ema_40_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close < lower KC + price < weekly EMA40 + volume spike
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema_40_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < EMA20 (middle of Keltner Channel)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > EMA20 (middle of Keltner Channel)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals