#!/usr/bin/env python3
# 1H_Stochastic_Pullback_4hTrend_Volume
# Hypothesis: 1h stochastic pullback in 4h trend direction with volume confirmation.
# Uses mean-reversion entries within established 4h trends, filtered by volume spikes.
# Works in bull/bear by following 4h trend direction; volume confirms institutional interest.
# Target: 15-37 trades/year per symbol (60-150 total over 4 years) to minimize fee drag.

name = "1H_Stochastic_Pullback_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')

    # Stochastic(14,3) on 1h
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values

    # Trend filter: 4h EMA50
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Volume confirmation: current volume > 2.0 x 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(k_percent[i]) or 
            np.isnan(d_percent[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Pullback to support in uptrend with volume spike
            if (k_percent[i] < 20 and 
                d_percent[i] < 20 and 
                close[i] > ema50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Pullback to resistance in downtrend with volume spike
            elif (k_percent[i] > 80 and 
                  d_percent[i] > 80 and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Stochastic overbought or trend turns down
            if k_percent[i] > 80 or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Stochastic oversold or trend turns up
            if k_percent[i] < 20 or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals