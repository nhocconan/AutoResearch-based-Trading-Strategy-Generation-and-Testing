#!/usr/bin/env python3
"""
4h_Choppiness_Filtered_Donchian_Breakout_With_Volume_Confirmation
Hypothesis: In choppy markets (high CHOP), Donchian breakouts fail; in trending markets (low CHOP), they succeed.
Use 1d ADX to filter regime: only trade when ADX > 25 (trending). Enter on Donchian(20) breakout with 1d volume > 1.5x 20-period average.
Exit on opposite Donchian(10) breakout for symmetry. Designed for 20-40 trades/year to avoid fee drag.
Works in both bull (trend up) and bear (trend down) markets via directional breakouts.
"""

name = "4h_Choppiness_Filtered_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate 1d ADX(14) for trend strength
    # +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align to same length

    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])

    # Smoothed values (Wilder smoothing = EMA with alpha=1/period)
    def wildeer_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # subsequent: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result

    tr_smooth = wildeer_smooth(tr, 14)
    plus_dm_smooth = wildeer_smooth(plus_dm, 14)
    minus_dm_smooth = wildeer_smooth(minus_dm, 14)

    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth

    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wildeer_smooth(dx, 14)  # ADX is smoothed DX

    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate Donchian channels on 4h
    # Donchian(20) for entry, Donchian(10) for exit
    def donchian_channel(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower

    donchian_20_upper, donchian_20_lower = donchian_channel(high, low, 20)
    donchian_10_upper, donchian_10_lower = donchian_channel(high, low, 10)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # warmup for indicators
        adx_val = adx_aligned[i]
        vol_avg_val = vol_avg_20_1d_aligned[i]

        if np.isnan(adx_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade when ADX > 25 (trending market)
        if adx_val > 25:
            if position == 0:
                # LONG: Price breaks above Donchian(20) upper + volume surge
                if high[i] > donchian_20_upper[i] and vol_1d[i // 16] > vol_avg_val * 1.5:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price breaks below Donchian(20) lower + volume surge
                elif low[i] < donchian_20_lower[i] and vol_1d[i // 16] > vol_avg_val * 1.5:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Price breaks below Donchian(10) lower
                if low[i] < donchian_10_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price breaks above Donchian(10) upper
                if high[i] > donchian_10_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In choppy market (ADX <= 25), stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals