#!/usr/bin/env python3
# 4h_1D_Camarilla_R4S4_Breakout_ADX_Volume
# Hypothesis: Breakouts at daily Camarilla R4/S4 levels with ADX trend filter and volume confirmation.
# Works in bull/bear: ADX > 25 filters chop, only trades strong trends.
# Entry: Price breaks R4/S4 with ADX>25 and volume spike (>2x 20-bar avg).
# Exit: Price re-enters R4/S4 or ADX drops below 20.
# Targets 20-40 trades/year to avoid fee drag. Focus on BTC/ETH.

name = "4h_1D_Camarilla_R4S4_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for ADX and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate ADX (14-period) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])

    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])

    # Smoothed TR, +DM, -DM (14-period Wilder smoothing)
    def wilders_smoothing(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) < period:
            return smoothed
        smoothed[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed

    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)

    # DI+ and DI-
    plus_di = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)

    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 14:
        adx[13] = np.nanmean(dx[:14])
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14

    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Get 1d data for Camarilla R4/S4 levels (from previous day)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Camarilla R4 and S4 levels (outer bands)
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2

    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # ADX trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # exit when trend weakens

        if position == 0:
            # LONG: Break above Camarilla R4 in strong trend with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and strong_trend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S4 in strong trend with volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and strong_trend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R4 or trend weakens
            if close[i] < camarilla_r4_aligned[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S4 or trend weakens
            if close[i] > camarilla_s4_aligned[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals