#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Trade 1-hour Camarilla pivot point breakouts filtered by 4-hour trend and 1-day volume.
# Long when price breaks above R1 with 4h uptrend and volume spike.
# Short when price breaks below S1 with 4h downtrend and volume spike.
# Exit when price retests the pivot point (PP) or trend reverses.
# Uses 4h trend filter to avoid counter-trend whipsaws and volume to confirm breakout strength.
# Designed for low trade frequency (15-35/year) to avoid fee drag, works in bull/bear via trend filter.

name = "1h_Camarilla_R1_S1_Breakout_Trend_Volume"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # Get 1d data for Camarilla pivot levels (previous day)
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels using previous day's OHLC
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    # Camarilla: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_pp = (c_high + c_low + c_close) / 3
    camarilla_r1 = c_close + (c_high - c_low) * 1.1 / 12
    camarilla_s1 = c_close - (c_high - c_low) * 1.1 / 12
    # Align to 1h: each 1d value applies to the next 24h (6 bars of 1h) after the day closes
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume filter: >1.5x 24-period average (6h)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):  # start after warmup for vol_avg_24
        # Skip if any required value is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg_24[i]) or \
           np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with 4h uptrend and volume spike
            if close[i] > camarilla_r1_aligned[i]:
                if close[i] > ema_50_4h_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                    signals[i] = 0.20
                    position = 1
            # SHORT: Price breaks below S1 with 4h downtrend and volume spike
            elif close[i] < camarilla_s1_aligned[i]:
                if close[i] < ema_50_4h_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                    signals[i] = -0.20
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests PP or trend turns down
            if close[i] < camarilla_pp_aligned[i]:  # retest pivot point
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_50_4h_aligned[i]:  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price retests PP or trend turns up
            if close[i] > camarilla_pp_aligned[i]:  # retest pivot point
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_50_4h_aligned[i]:  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals