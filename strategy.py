#!/usr/bin/env python3
# 1d_Camarilla_Pivot_R1_S1_Breakout_WeeklyTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from daily candles act as support/resistance.
# Long when price breaks above R1 with volume spike and weekly uptrend.
# Short when price breaks below S1 with volume spike and weekly downtrend.
# Uses weekly EMA20 for trend filter and volume > 1.5x 20-period average for confirmation.
# Designed for 1d timeframe to avoid overtrading. Works in bull markets via breakouts in uptrends
# and in bear markets via breakdowns in downtrends. Low trade frequency (~10-25/year) minimizes fee drag.

name = "1d_Camarilla_Pivot_R1_S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Daily Camarilla pivot levels (R1, S1)
    # Calculated from previous day's OHLC
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]

    # Pivot point
    pivot = (high_shift + low_shift + close_shift) / 3.0
    # R1 and S1 levels
    r1 = pivot + (high_shift - low_shift) * 1.1 / 12
    s1 = pivot - (high_shift - low_shift) * 1.1 / 12

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ok[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(pivot[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]

        if position == 0:
            # LONG: Price breaks above R1 with volume spike and weekly uptrend
            if close[i] > r1[i] and volume_ok[i] and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and weekly downtrend
            elif close[i] < s1[i] and volume_ok[i] and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below pivot or trend turns down
            if close[i] < pivot[i] or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above pivot or trend turns up
            if close[i] > pivot[i] or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals