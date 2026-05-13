#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot points on 1h for entry timing, with daily trend and volume confirmation.
# The Camarilla R1/S1 levels provide tight breakout zones. Only trade in direction of daily trend.
# Volume spike confirms breakout strength. Works in bull/bear by following daily trend.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).
# Uses 1h for entry timing, daily for trend filter (higher timeframe = fewer, higher quality signals).

name = "1h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "1h"
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

    # Get daily data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d

    # Calculate Camarilla pivot points for 1h (using previous 1h bar's OHLC)
    # Camarilla formulas: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We need previous bar's OHLC, so shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar will have invalid values (rolled from last), set to 0 so range is 0
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]

    hl_range = prev_high - prev_low
    camarilla_R1 = prev_close + 1.1 * hl_range / 12
    camarilla_S1 = prev_close - 1.1 * hl_range / 12

    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to have valid previous bar data
        # Get aligned daily values for current 1h bar
        if i < len(daily_uptrend):
            up_trend = daily_uptrend[i]
            down_trend = daily_downtrend[i]
        else:
            up_trend = False
            down_trend = False

        vol_spike = volume_spike[i] if i < len(volume_spike) else False
        in_sess = in_session[i]

        if not in_sess:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Daily uptrend + price breaks above Camarilla R1 + volume spike
            if (up_trend and 
                close[i] > camarilla_R1[i] and vol_spike):
                signals[i] = 0.20
                position = 1
            # SHORT: Daily downtrend + price breaks below Camarilla S1 + volume spike
            elif (down_trend and 
                  close[i] < camarilla_S1[i] and vol_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 or trend changes
            if (close[i] < camarilla_S1[i] or not up_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 or trend changes
            if (close[i] > camarilla_R1[i] or not down_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals