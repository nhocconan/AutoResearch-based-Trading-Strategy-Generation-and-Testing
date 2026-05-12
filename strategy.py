#!/usr/bin/env python3
# 1d_TRIX_VolumeSpike_1wTrend
# Hypothesis: TRIX momentum with volume spike confirmation and weekly trend filter.
# TRIX filters noise and identifies momentum; volume spike confirms conviction.
# Weekly trend filter ensures trades align with long-term direction, reducing false signals in chop.
# Designed for 10-25 trades/year per symbol, works in both bull and bear markets via trend alignment.

name = "1d_TRIX_VolumeSpike_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    # Weekly EMA34 trend filter (smooth, reliable trend)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Calculate TRIX on daily close: EMA of EMA of EMA of log returns, then ROC
    # TRIX(18) = 100 * (EMA3 of log(close) ROC)
    log_close = np.log(close)
    ema1 = pd.Series(log_close).ewm(span=18, adjust=False, min_periods=18).mean().values
    ema2 = pd.Series(ema1).ewm(span=18, adjust=False, min_periods=18).mean().values
    ema3 = pd.Series(ema2).ewm(span=18, adjust=False, min_periods=18).mean().values
    trix = 100 * (pd.Series(ema3).pct_change(1).values)  # ROC of triple EMA

    # Volume confirmation: current volume > 2.0x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA34
        price_above_weekly_ema = close[i] > ema_34_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_34_1w_aligned[i]

        if position == 0:
            # LONG: TRIX turns positive AND above weekly EMA34 AND volume spike
            if trix[i] > 0 and trix[i] > trix[i-1] and price_above_weekly_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX turns negative AND below weekly EMA34 AND volume spike
            elif trix[i] < 0 and trix[i] < trix[i-1] and price_below_weekly_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative OR weekly trend turns down
            if trix[i] < 0 or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive OR weekly trend turns up
            if trix[i] > 0 or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals