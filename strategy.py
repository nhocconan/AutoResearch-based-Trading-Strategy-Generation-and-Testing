#!/usr/bin/env python3

# 4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike
# Hypothesis: Use 1-day Camarilla R1/S1 levels as breakout triggers with 1-day EMA34 trend filter and volume spike confirmation.
# The Camarilla pivot system identifies key support/resistance levels where price often reverses or breaks out.
# In trending markets, breaks of R1/S1 with volume and trend alignment capture strong moves.
# Works in both bull and bear markets by requiring trend alignment and volume confirmation to avoid false signals.
# Target: 25-40 trades/year (~100-160 total over 4 years).

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
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

    # Get 1d data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1-day Camarilla levels (based on previous day's OHLC)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C, H, L are from previous completed day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day (based on previous day)
    camarilla_R1 = np.full(len(df_1d), np.nan)
    camarilla_S1 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        camarilla_R1[i] = prev_close + (prev_high - prev_low) * 1.1 / 12
        camarilla_S1[i] = prev_close - (prev_high - prev_low) * 1.1 / 12

    # Align Camarilla levels to 4h timeframe (wait for day to complete)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)

    # Calculate 1-day EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Volume confirmation: current volume > 2.0x average of last 20 periods (strict)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup for volume MA
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above R1 with bullish trend and volume spike
            if close[i] > camarilla_R1_aligned[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with bearish trend and volume spike
            elif close[i] < camarilla_S1_aligned[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (opposite level) or trend turns bearish
            if close[i] < camarilla_S1_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (opposite level) or trend turns bullish
            if close[i] > camarilla_R1_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals