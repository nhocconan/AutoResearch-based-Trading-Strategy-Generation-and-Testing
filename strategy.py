#!/usr/bin/env python3
# 12h_Woody_CCI_ZeroReject_Trend
# Hypothesis: Uses Woodies CCI with zero-reject method on 12h timeframe combined with 1-week EMA trend filter and volume confirmation.
# In ranging markets, CCI crosses zero with momentum; in trending markets, price respects weekly EMA.
# Long: CCI crosses above zero + price > weekly EMA + volume spike. Short: CCI crosses below zero + price < weekly EMA + volume spike.
# Exit: CCI returns to zero (mean reversion) to avoid overstaying.
# Target: 12-30 trades/year on 12h to stay within optimal range while capturing momentum in both bull and bear markets.

name = "12h_Woody_CCI_ZeroReject_Trend"
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

    # Woodies CCI (20-period)
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci = (typical_price - sma_tp) / (0.015 * mad)

    # Weekly EMA for trend filter (from 1w data)
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(cci[i]) or np.isnan(cci[i-1]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CCI crosses above zero + price > weekly EMA + volume spike
            if (cci[i-1] <= 0 and cci[i] > 0 and 
                close[i] > ema_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: CCI crosses below zero + price < weekly EMA + volume spike
            elif (cci[i-1] >= 0 and cci[i] < 0 and 
                  close[i] < ema_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CCI returns to zero (mean reversion)
            if cci[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CCI returns to zero (mean reversion)
            if cci[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals