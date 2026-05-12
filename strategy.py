#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance levels.
# Breakouts above R1 level with volume confirmation and daily trend alignment go long.
# Breakdowns below S1 level with volume confirmation and daily trend alignment go short.
# Uses tight entry conditions to maintain 20-50 trades/year, avoiding fee drag.
# Works in both bull and bear markets by trading breakouts of significant daily levels with trend filter.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)

    # Calculate Camarilla pivot levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: Pivot = (H+L+C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # R1 = Pivot + (Range * 1.1/12)
    # S1 = Pivot - (Range * 1.1/12)
    r1 = pivot + (range_1d * 1.1 / 12.0)
    s1 = pivot - (range_1d * 1.1 / 12.0)
    
    # Camarilla levels need only the completed daily candle
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Daily EMA34 trend filter (only needs completed daily candle)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA34
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above R1 level with volume and uptrend
            if (not np.isnan(r1_aligned[i]) and 
                close[i] > r1_aligned[i] and
                price_above_daily_ema and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 level with volume and downtrend
            elif (not np.isnan(s1_aligned[i]) and 
                  close[i] < s1_aligned[i] and
                  price_below_daily_ema and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 level or trend turns down
            if (not np.isnan(s1_aligned[i]) and 
                close[i] < s1_aligned[i]) or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 level or trend turns up
            if (not np.isnan(r1_aligned[i]) and 
                close[i] > r1_aligned[i]) or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals