#!/usr/bin/env python3
# 4h_1D_Camarilla_R1_S1_Breakout_Trend_VolumeS_v3
# Hypothesis: Refined version with reduced trade frequency by requiring volume > 2.0x average volume (stricter filter) and adding ADX(14) > 25 as trend strength filter. This should reduce trades to 50-100/year while maintaining edge in both bull and bear markets by combining Camarilla breakouts with strong trend confirmation and volume validation.

name = "4h_1D_Camarilla_R1_S1_Breakout_Trend_VolumeS_v3"
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

    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels (R1, S1) from previous day
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12

    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 1d ADX(14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1]) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(0, low[i-1] - low[i]) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        dx = np.zeros_like(tr)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_sum = np.sum(plus_dm[1:period])
        minus_dm_sum = np.sum(minus_dm[1:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # Smooth DX to get ADX
        adx = np.full_like(tr, np.nan)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(tr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx

    adx14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx14_1d)

    # Calculate 4h volume SMA20 for volume confirmation (with spike filter > 2.0x)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0  # Require 2.0x average volume (stricter)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i]) or
            np.isnan(adx14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R1 in 1d uptrend with volume spike and ADX > 25
            if (close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and 
                volume[i] > volume_spike_threshold[i] and adx14_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 in 1d downtrend with volume spike and ADX > 25
            elif (close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and 
                  volume[i] > volume_spike_threshold[i] and adx14_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals