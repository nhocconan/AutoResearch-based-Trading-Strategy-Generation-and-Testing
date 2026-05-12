#!/usr/bin/env python3
# 4h_RSI_MeanReversion_VolumeSpike_1dTrend
# Hypothesis: Mean reversion on 4h using RSI(14) with volume spike confirmation and 1d trend filter.
# Long when RSI < 30 (oversold), volume > 2x 20-period average, and price above 1d EMA50.
# Short when RSI > 70 (overbought), volume > 2x 20-period average, and price below 1d EMA50.
# Exits when RSI returns to neutral (40-60) or opposite extreme is reached.
# Designed for 4h to avoid overtrading. Works in both bull and bear markets via mean reversion.

name = "4h_RSI_MeanReversion_VolumeSpike_1dTrend"
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

    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: current volume > 2x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

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
            # LONG: RSI < 30 (oversold), volume confirmation, price above daily EMA50
            if (rsi[i] < 30 and volume_ok[i] and price_above_daily_ema):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought), volume confirmation, price below daily EMA50
            elif (rsi[i] > 70 and volume_ok[i] and price_below_daily_ema):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>=40) or overbought (>70)
            if rsi[i] >= 40 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<=60) or oversold (<30)
            if rsi[i] <= 60 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals