#!/usr/bin/env python3
# 1d_WeeklyTrend_RSI_Reversal_With_Volume
# Hypothesis: Use weekly RSI extremes (>70 or <30) for trend direction, then look for daily RSI reversals from overbought/oversold levels with volume confirmation. In bull markets, buy dips in uptrend; in bear markets, sell rallies in downtrend. Weekly trend filter prevents counter-trend trades, while daily RSI provides precise entry timing. Volume confirmation ensures institutional participation. Designed for 7-25 trades/year per symbol, works in both bull and bear via trend-aligned mean reversion.

name = "1d_WeeklyTrend_RSI_Reversal_With_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)

    # Weekly RSI(14) for trend filter
    delta = pd.Series(df_1w['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_14_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)

    # Daily RSI(14) for entry signals
    delta_d = pd.Series(close).diff()
    gain_d = (delta_d.where(delta_d > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss_d = (-delta_d.where(delta_d < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs_d = gain_d / loss_d
    rsi_14_d = 100 - (100 / (1 + rs_d))
    rsi_d = rsi_14_d.values

    # Volume confirmation: current volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_d[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter: RSI > 50 = uptrend, RSI < 50 = downtrend
        weekly_uptrend = rsi_1w_aligned[i] > 50
        weekly_downtrend = rsi_1w_aligned[i] < 50

        if position == 0:
            # LONG: Weekly uptrend AND daily RSI oversold (<30) AND volume
            if weekly_uptrend and rsi_d[i] < 30 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend AND daily RSI overbought (>70) AND volume
            elif weekly_downtrend and rsi_d[i] > 70 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Daily RSI overbought (>70) OR trend turns down
            if rsi_d[i] > 70 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Daily RSI oversold (<30) OR trend turns up
            if rsi_d[i] < 30 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals