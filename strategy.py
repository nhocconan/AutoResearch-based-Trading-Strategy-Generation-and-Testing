#!/usr/bin/env python3

# 1H_4H_1D_Camarilla_R1S1_Breakout_Trend_Volume
# Hypothesis: Use 4h/1d for trend direction (EMA), 1h for entry timing at Camarilla R1/S1 breakouts with volume confirmation.
# 4h EMA34 defines trend, 1d EMA34 confirms higher timeframe bias. Enter long when price breaks above R1 with bullish 4h trend and volume spike.
# Enter short when price breaks below S1 with bearish 4h trend and volume spike. Exit on opposite signal or trend flip.
# Session filter (08-20 UTC) reduces noise. Target 15-37 trades/year to avoid fee drag.

name = "1H_4H_1D_Camarilla_R1S1_Breakout_Trend_Volume"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # Get 1d data for higher timeframe trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Calculate 1d EMA for higher timeframe trend confirmation
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate daily Camarilla levels (based on previous day)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use previous day's high, low, close to calculate today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12

    # Volume confirmation: current volume > 1.5x average of last 24 hours
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_ok = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(session_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filters
        bullish_4h = close[i] > ema_4h_aligned[i]
        bearish_4h = close[i] < ema_4h_aligned[i]
        bullish_1d = close[i] > ema_1d_aligned[i]
        bearish_1d = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price crosses above R1 with bullish 4h trend, bullish 1d confirmation, volume and session
            if (close[i] > R1[i] and close[i-1] <= R1[i-1] and 
                bullish_4h and bullish_1d and volume_ok[i] and session_ok[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price crosses below S1 with bearish 4h trend, bearish 1d confirmation, volume and session
            elif (close[i] < S1[i] and close[i-1] >= S1[i-1] and 
                  bearish_4h and bearish_1d and volume_ok[i] and session_ok[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or 4h trend turns bearish or session ends
            if (close[i] < S1[i] and close[i-1] >= S1[i-1]) or (not bullish_4h) or (not session_ok[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or 4h trend turns bullish or session ends
            if (close[i] > R1[i] and close[i-1] <= R1[i-1]) or (not bearish_4h) or (not session_ok[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals