#!/usr/bin/env python3

# 1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume
# Hypothesis: Use 4h/1d trend direction and daily Camarilla levels for entry timing on 1h.
# Long when 1h price crosses above daily R1 with 4h/1d uptrend and volume confirmation.
# Short when 1h price crosses below daily S1 with 4h/1d downtrend and volume confirmation.
# Uses 1h for entry timing only to reduce trade frequency. Target: 15-30 trades/year.
# Works in bull/bear via trend filter and volatility-adjusted position sizing.

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "1h"
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

    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4h EMA50 for additional trend confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Calculate daily Camarilla levels (based on previous day)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Multi-timeframe trend filter: both 4h and 1d must agree
        bullish_trend = (close[i] > ema_4h_aligned[i]) and (close[i] > ema_1d_aligned[i])
        bearish_trend = (close[i] < ema_4h_aligned[i]) and (close[i] < ema_1d_aligned[i])

        if position == 0:
            # LONG: Price crosses above R1 with bullish 4h/1d trend and volume confirmation
            if close[i] > R1[i] and close[i-1] <= R1[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price crosses below S1 with bearish 4h/1d trend and volume confirmation
            elif close[i] < S1[i] and close[i-1] >= S1[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or 4h/1d trend turns bearish
            if close[i] < S1[i] and close[i-1] >= S1[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or 4h/1d trend turns bullish
            if close[i] > R1[i] and close[i-1] <= R1[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals