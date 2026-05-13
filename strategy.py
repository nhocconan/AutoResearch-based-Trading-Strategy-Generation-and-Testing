#!/usr/bin/env python3
# 6h_HeikinAshi_TRIX_Trend_With_1d_Volume_Spike_Filter
# Hypothesis: Combines Heikin-Ashi smoothed candles with TRIX momentum to filter noise. 
# Long when HA close > HA open AND TRIX rising; short when HA close < HA open AND TRIX falling.
# Uses 1d volume spike (>2x 20-period average) to confirm institutional interest and avoid chop.
# Works in bull via trend continuation and bear via mean reversion spikes. Target: 15-30 trades/year.

name = "6h_HeikinAshi_TRIX_Trend_With_1d_Volume_Spike_Filter"
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
    open_price = prices['open'].values

    # Heikin-Ashi calculation
    ha_close = (open_price + high + low + close) / 4
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (open_price[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

    # TRIX: Triple EMA of ROC (1-period)
    roc = np.diff(close, prepend=close[0]) / close  # 1-period ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3  # TRIX value

    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_avg_20 * 2.0)  # True when volume > 2x average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(15, n):  # Start after TRIX warmup
        # Skip if any required value is NaN
        if (np.isnan(ha_open[i]) or np.isnan(ha_close[i]) or 
            np.isnan(trix[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: HA bullish (close > open) AND TRIX rising AND volume spike
            if (ha_close[i] > ha_open[i] and 
                trix[i] > trix[i-1] and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: HA bearish (close < open) AND TRIX falling AND volume spike
            elif (ha_close[i] < ha_open[i] and 
                  trix[i] < trix[i-1] and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: HA bearish reversal OR TRIX turns down
            if (ha_close[i] < ha_open[i] or trix[i] < trix[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: HA bullish reversal OR TRIX turns up
            if (ha_close[i] > ha_open[i] or trix[i] > trix[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals