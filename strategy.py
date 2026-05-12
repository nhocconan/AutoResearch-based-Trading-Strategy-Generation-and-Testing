#!/usr/bin/env python3

# 12h_1W_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: Breakout above Camarilla R3 or below S3 on 12h with 1w trend filter and volume confirmation.
# Camarilla levels provide statistically significant support/resistance. Works in both bull and bear markets
# by requiring trend alignment from higher timeframe and volume confirmation to avoid false breakouts.
# Targets 15-30 trades/year on 12h timeframe.

name = "12h_1W_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "12h"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = Close + ((High - Low) * 1.1/2), R3 = Close + ((High - Low) * 1.1/4)
    #          S3 = Close - ((High - Low) * 1.1/4), S4 = Close - ((High - Low) * 1.1/2)
    # We need previous day's OHLC, so we shift by 2 bars (since 12h bars, 2 bars = 1 day)
    if len(high) < 2:
        return np.zeros(n)
    
    prev_high = np.roll(high, 2)
    prev_low = np.roll(low, 2)
    prev_close = np.roll(close, 2)
    
    # Handle first two bars
    prev_high[:2] = 0
    prev_low[:2] = 0
    prev_close[:2] = 0
    
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + (rang * 1.1 / 4)
    camarilla_s3 = prev_close - (rang * 1.1 / 4)

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(2, n):  # Start from 2 to have valid Camarilla levels
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1w
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R3 with bullish trend and volume confirmation
            if close[i] > camarilla_r3[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S3 with bearish trend and volume confirmation
            elif close[i] < camarilla_s3[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla S3 or trend turns bearish
            if close[i] < camarilla_s3[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla R3 or trend turns bullish
            if close[i] > camarilla_r3[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals