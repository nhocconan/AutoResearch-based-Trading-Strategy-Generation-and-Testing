#!/usr/bin/env python3
# 6h_OrderBlock_Retest_Volume
# Hypothesis: Identify institutional order blocks from 1d candles (bullish: close > open, bearish: close < open) and wait for price to retest these zones on 6h with volume confirmation. Enter long when price retraces to a bullish OB with volume > 1.5x 20-period average; short when price retraces to a bearish OB with volume spike. Exit when price moves beyond the OB opposite side. Targets 15-35 trades/year to avoid fee drag and works in both bull/bear via retest logic in ranging/trending markets.

name = "6h_OrderBlock_Retest_Volume"
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

    # Get 1d data for order blocks
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values

    # Identify bullish and bearish order blocks on 1d
    bullish_ob = close_1d > open_1d  # bullish candle
    bearish_ob = close_1d < open_1d  # bearish candle

    # For bullish OB: use low as support zone
    ob_bullish_low = np.where(bullish_ob, low_1d, np.nan)
    # For bearish OB: use high as resistance zone
    ob_bearish_high = np.where(bearish_ob, high_1d, np.nan)

    # Align OB levels to 6h timeframe (wait for 1d close)
    ob_bullish_low_aligned = align_htf_to_ltf(prices, df_1d, ob_bullish_low)
    ob_bearish_high_aligned = align_htf_to_ltf(prices, df_1d, ob_bearish_high)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if required values are NaN
        if (np.isnan(ob_bullish_low_aligned[i]) or np.isnan(ob_bearish_high_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price retraces to bullish OB (support) with volume spike
            if (not np.isnan(ob_bullish_low_aligned[i]) and
                low[i] <= ob_bullish_low_aligned[i] * 1.005 and  # allow 0.5% slippage
                close[i] > ob_bullish_low_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price retraces to bearish OB (resistance) with volume spike
            elif (not np.isnan(ob_bearish_high_aligned[i]) and
                  high[i] >= ob_bearish_high_aligned[i] * 0.995 and  # allow 0.5% slippage
                  close[i] < ob_bearish_high_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below OB support
            if not np.isnan(ob_bullish_low_aligned[i]) and close[i] < ob_bullish_low_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above OB resistance
            if not np.isnan(ob_bearish_high_aligned[i]) and close[i] > ob_bearish_high_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals