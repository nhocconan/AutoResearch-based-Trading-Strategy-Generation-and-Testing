#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot breakouts at R1/S1 levels with daily trend filter and volume spike
# capture institutional order flow with low whipsaw. Works in bull markets (breakouts above R1 in uptrend)
# and bear markets (breakdowns below S1 in downtrend). Daily trend ensures alignment with higher-timeframe
# momentum, reducing false signals. Volume filter confirms breakout strength. Target: 20-40 trades/year.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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

    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate daily Camarilla levels: R1, S1
    # Formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12

    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 with volume spike and daily uptrend
            if close[i] > camarilla_r1_aligned[i] and volume_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 with volume spike and daily downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla pivot point (S1) or daily trend turns down
            # Pivot point = (H+L+C)/3
            daily_idx = i // 6  # 6x 4h bars in 1 day
            if daily_idx < len(daily_high):
                pivot_point = (daily_high[daily_idx] + daily_low[daily_idx] + daily_close[daily_idx]) / 3
                if close[i] < pivot_point or close[i] < ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla pivot point (R1) or daily trend turns up
            daily_idx = i // 6  # 6x 4h bars in 1 day
            if daily_idx < len(daily_high):
                pivot_point = (daily_high[daily_idx] + daily_low[daily_idx] + daily_close[daily_idx]) / 3
                if close[i] > pivot_point or close[i] > ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25

    return signals