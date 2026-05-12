#!/usr/bin/env python3
# 6h_12h_ADX25_EMA50_Trend
# Hypothesis: Trade in the direction of the 12h EMA50 trend only when 12h ADX > 25 (strong trend).
# Enter on 6h close crossing above/below 6h EMA20 with momentum confirmation (RSI > 50 for long, < 50 for short).
# Exit when trend weakens (ADX < 20) or price crosses back over EMA20.
# Designed for 12-30 trades/year on 6h timeframe. Works in bull markets by following uptrends,
# and in bear markets by following downtrends, while avoiding range-bound whipsaws via ADX filter.

name = "6h_12h_ADX25_EMA50_Trend"
timeframe = "6h"
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

    # Get 12h data for trend filter (EMA50) and regime filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # 12h ADX for trend strength
    # Calculate True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

    # Calculate Directional Movement
    up_move = df_12h['high'].diff()
    down_move = df_12h['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed DM and TR
    plus_di_12h = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_12h
    dx_12h = 100 * abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False).mean().values

    # Align 12h indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)

    # 6h EMA20 for entry timing
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # 6h RSI for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(ema_20_6h[i]) or np.isnan(rsi_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend and regime filters
        bullish_trend = close[i] > ema_50_12h_aligned[i]
        bearish_trend = close[i] < ema_50_12h_aligned[i]
        strong_trend = adx_12h_aligned[i] > 25
        weak_trend = adx_12h_aligned[i] < 20

        if position == 0:
            # LONG: Price crosses above EMA20 with bullish trend, strong ADX, and bullish momentum
            if (close[i] > ema_20_6h[i] and close[i-1] <= ema_20_6h[i-1] and
                bullish_trend and strong_trend and rsi_6h[i] > 50):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below EMA20 with bearish trend, strong ADX, and bearish momentum
            elif (close[i] < ema_20_6h[i] and close[i-1] >= ema_20_6h[i-1] and
                  bearish_trend and strong_trend and rsi_6h[i] < 50):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakens or price crosses back below EMA20
            if weak_trend or (close[i] < ema_20_6h[i] and close[i-1] >= ema_20_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens or price crosses back above EMA20
            if weak_trend or (close[i] > ema_20_6h[i] and close[i-1] <= ema_20_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals