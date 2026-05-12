#!/usr/bin/env python3
# 1h_CCI_OverboughtOversold_4hTrend_Filter
# Hypothesis: Use 1-hour CCI for overbought/oversold reversal signals, filtered by 4-hour trend direction.
# In trending markets (4h), counter-trend CCI reversals offer high-probability mean-reversion entries.
# Works in bull markets via long entries on 4h uptrend + CCI oversold.
# Works in bear markets via short entries on 4h downtrend + CCI overbought.
# Added 4h volume spike filter to confirm institutional participation and reduce whipsaws.
# Target: 20-50 trades/year by using strict CCI thresholds and trend alignment.

name = "1h_CCI_OverboughtOversold_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')

    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # Calculate 4h volume spike: current > 1.8x average of last 6 bars (~1 day)
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=6, min_periods=6).mean().values
    volume_spike_4h = volume_4h > (1.8 * vol_ma_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)

    # CCI (20-period) on 1h
    tp = (high + low + close) / 3.0  # typical price
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(np.abs(tp - ma_tp)).rolling(window=20, min_periods=20).mean().values
    cci = (tp - ma_tp) / (0.015 * mad)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after CCI warmup
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(cci[i]) or 
            np.isnan(volume_spike_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h UPTREND + CCI OVERSOLD (< -100) + 4h VOLUME SPIKE
            if (close[i] > ema_50_4h_aligned[i] and 
                cci[i] < -100 and 
                volume_spike_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h DOWNTREND + CCI OVERBOUGHT (> 100) + 4h VOLUME SPIKE
            elif (close[i] < ema_50_4h_aligned[i] and 
                  cci[i] > 100 and 
                  volume_spike_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CCI crosses above zero or trend breaks
            if cci[i] > 0 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: CCI crosses below zero or trend breaks
            if cci[i] < 0 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals