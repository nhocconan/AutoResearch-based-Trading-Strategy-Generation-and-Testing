#!/usr/bin/env python3
# 1h_4H1D_Trend_With_Volume_Confirmation
# Hypothesis: Use 4h trend direction (EMA50) and 1d momentum (RSI > 50) for signal direction,
# Enter on 1h pullbacks to EMA21 with volume spike (>1.5x 20-bar average) during 08-20 UTC.
# Exit on opposite 4h EMA50 cross. Designed for low turnover (15-35/year) to avoid fee drag.

name = "1h_4H1D_Trend_With_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)

    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Get 1d data for momentum filter (RSI(14) > 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # 1h EMA21 for entry timing
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h uptrend (price > EMA50), 1d bullish (RSI > 50), volume spike, pullback to EMA21
            if (close[i] > ema_4h_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                volume_spike[i] and 
                close[i] <= ema_21[i] * 1.005 and  # Allow small buffer above EMA21
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend (price < EMA50), 1d bearish (RSI < 50), volume spike, pullback to EMA21
            elif (close[i] < ema_4h_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  volume_spike[i] and 
                  close[i] >= ema_21[i] * 0.995 and  # Allow small buffer below EMA21
                  in_session[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h trend turns down (price < EMA50)
            if close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h trend turns up (price > EMA50)
            if close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals