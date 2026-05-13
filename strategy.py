#!/usr/bin/env python3
# 4h_Trix_PriceAction_TrendVol
# Hypothesis: TRIX (12) signals momentum direction; price action confirms via
# higher highs/lows; volume surge filters false signals. Works in bull (long
# momentum) and bear (short momentum) with 1d trend filter to avoid counter-
# trend trades. Target: 20-40 trades/year.

name = "4h_Trix_PriceAction_TrendVol"
timeframe = "4h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # TRIX (12) on close
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values

    # Volume spike: current > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Price action: HH/HL for long, LH/LL for short
    hh = high > np.maximum.accumulate(high)
    ll = low < np.minimum.accumulate(low)
    # Require confirmation over last 3 bars
    hh_confirm = hh & np.roll(hh, 1) & np.roll(hh, 2)
    ll_confirm = ll & np.roll(ll, 1) & np.roll(ll, 2)

    # 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX up, price making HH/HL, volume spike, 1d uptrend
            if (trix[i] > 0 and trix[i] > trix[i-1] and
                hh_confirm[i] and volume_spike[i] and
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX down, price making LH/LL, volume spike, 1d downtrend
            elif (trix[i] < 0 and trix[i] < trix[i-1] and
                  ll_confirm[i] and volume_spike[i] and
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns down or price makes LL
            if trix[i] < 0 or ll_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns up or price makes HH
            if trix[i] > 0 or hh_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals