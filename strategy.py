#!/usr/bin/env python3
"""
6h_ConnorRSI_2_Breakout_1dTrend_Volume
Hypothesis: Connor RSI (RSI(3) + RSI of streak + percent rank) identifies extreme mean-reversion points. 
Long when Connors RSI < 10 and price > 1d EMA50 (uptrend filter), short when Connors RSI > 90 and price < 1d EMA50.
Volume confirmation (6h volume > 1.5x 20-period average) filters false signals. 
Designed for 12-30 trades/year on 6h timeframe to work in both bull and bear markets via mean reversion with trend filter.
"""

name = "6h_ConnorRSI_2_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate Connor RSI components
    # RSI(3)
    delta = pd.Series(close).diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = pd.Series(up).ewm(span=3, adjust=False, min_periods=3).mean()
    ma_down = pd.Series(down).ewm(span=3, adjust=False, min_periods=3).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    rsi_vals = rsi.values

    # Streak RSI: count consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    streak = np.where(streak > 0, streak, np.where(streak < 0, -streak, 0))  # absolute streak length
    # RSI of streak (using 2-period as per Connors)
    streak_change = pd.Series(streak).diff()
    streak_up = streak_change.clip(lower=0)
    streak_down = -streak_change.clip(upper=0)
    streak_ma_up = pd.Series(streak_up).ewm(span=2, adjust=False, min_periods=2).mean()
    streak_ma_down = pd.Series(streak_down).ewm(span=2, adjust=False, min_periods=2).mean()
    streak_rsi = 100 - (100 / (1 + streak_ma_up / streak_ma_down))
    streak_rsi_vals = streak_rsi.values

    # Percent Rank: percentage of values in lookback period that are below current value
    lookback = 100
    percent_rank = np.zeros(len(close))
    for i in range(lookback, len(close)):
        window = close[i-lookback:i]
        percent_rank[i] = (np.sum(window < close[i]) / lookback) * 100
    # For initial period, use expanding window
    for i in range(lookback):
        window = close[:i+1]
        if len(window) > 0:
            percent_rank[i] = (np.sum(window < close[i]) / len(window)) * 100

    # Connor RSI = (RSI(3) + Streak RSI(2) + PercentRank(100)) / 3
    connors_rsi = (rsi_vals + streak_rsi_vals + percent_rank) / 3.0

    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        connors_val = connors_rsi[i]
        ema50_val = ema50_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(connors_val) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Connors RSI < 10 (oversold) + uptrend + volume confirmation
            if connors_val < 10 and close[i] > ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Connors RSI > 90 (overbought) + downtrend + volume confirmation
            elif connors_val > 90 and close[i] < ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Connors RSI > 50 (mean reversion) or trend breakdown
            if connors_val > 50 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Connors RSI < 50 (mean reversion) or trend reversal
            if connors_val < 50 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals