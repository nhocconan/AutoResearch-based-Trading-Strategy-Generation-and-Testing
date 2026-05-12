#!/usr/bin/env python3
"""
1h_OrderBookImbalance_Reversal_4hTrend
Hypothesis: Intraday reversals occur when order book imbalance (proxied by VWAP deviation) reaches extremes while higher timeframe trend remains intact. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets by combining mean reversion with trend filter.
"""

name = "1h_OrderBookImbalance_Reversal_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # VWAP approximation: (high + low + close) / 3 * volume
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    vwap = np.where(volume.cumsum() == 0, typical_price, vwap)

    # Deviation from VWAP as % - proxy for order book imbalance
    deviation = (close - vwap) / vwap * 100

    # Z-score of deviation over 20 periods
    dev_mean = pd.Series(deviation).rolling(window=20, min_periods=20).mean().values
    dev_std = pd.Series(deviation).rolling(window=20, min_periods=20).std().values
    z_score = np.where(dev_std != 0, (deviation - dev_mean) / dev_std, 0)

    # Get 4h data for trend filter (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    # 4h EMA20 for trend
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(ema20_4h_aligned[i]) or np.isnan(z_score[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Extreme negative deviation (oversold) + 4h uptrend
            if z_score[i] < -2.0 and close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Extreme positive deviation (overbought) + 4h downtrend
            elif z_score[i] > 2.0 and close[i] < ema20_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Deviation returns to mean or 4h trend turns down
            if abs(z_score[i]) < 0.5 or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Deviation returns to mean or 4h trend turns up
            if abs(z_score[i]) < 0.5 or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals