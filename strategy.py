#!/usr/bin/env python3
# 1d_Weekly_Keltner_Channel_Breakout_Trend_Filter
# Hypothesis: Use weekly Keltner Channel breakout with daily trend filter to capture strong trends in BTC/ETH.
# Combines volatility-based breakout (Keltner) with weekly trend alignment to avoid false signals.
# Works in both bull and bear markets by following the weekly trend direction.
# Target: 15-25 trades/year (60-100 total) to minimize fee drag while maintaining edge.

name = "1d_Weekly_Keltner_Channel_Breakout_Trend_Filter"
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

    # Get weekly data for trend and Keltner calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)

    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values

    # Calculate weekly ATR(10) for Keltner Channel
    tr1 = weekly_high[1:] - weekly_low[1:]
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Weekly EMA(20) for trend filter and Keltner center
    ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner Channel: EMA(20) ± 2 * ATR(10)
    keltner_upper = ema_20 + 2 * atr_10
    keltner_lower = ema_20 - 2 * atr_10

    # Align weekly indicators to daily timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_weekly, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_weekly, keltner_lower)
    ema_20_aligned = align_htf_to_ltf(prices, df_weekly, ema_20)

    # Daily volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly Keltner upper + weekly uptrend + volume spike
            if (close[i] > keltner_upper_aligned[i] and 
                close[i] > ema_20_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Keltner lower + weekly downtrend + volume spike
            elif (close[i] < keltner_lower_aligned[i] and 
                  close[i] < ema_20_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly EMA(20) or re-enters Keltner Channel
            if close[i] < ema_20_aligned[i] or close[i] < keltner_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly EMA(20) or re-enters Keltner Channel
            if close[i] > ema_20_aligned[i] or close[i] > keltner_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals