#!/usr/bin/env python3
# 4h_RSI_Trend_Pullback_v1
# Hypothesis: In trending markets (identified by daily EMA50), RSI(14) pullbacks to 40-60 range offer high-probability entries. Long when price > daily EMA50 and RSI crosses above 40 from below; short when price < daily EMA50 and RSI crosses below 60 from above. Volume confirmation (>1.5x 20-period average) filters low-conviction moves. Designed for 4h to limit trades (~25-40/year) and avoid fee drag. Works in bull via pullbacks in uptrends and in bear via bounces in downtrends.

name = "4h_RSI_Trend_Pullback_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA50
        price_above_daily_ema = close[i] > ema_50_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: price above daily EMA50 and RSI crosses above 40 from below with volume
            if i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] <= 40 and rsi[i] > 40 and volume_ok[i] and price_above_daily_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: price below daily EMA50 and RSI crosses below 60 from above with volume
            elif i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] >= 60 and rsi[i] < 60 and volume_ok[i] and price_below_daily_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below 40 or trend turns down
            if i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] >= 40 and rsi[i] < 40 or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above 60 or trend turns up
            if i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] <= 60 and rsi[i] > 60 or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals