#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend
Hypothesis: Price breaking above/below 4h Donchian(20) channels with volume > 1.5x average and 4h EMA(50) trend filter captures strong trending moves. Uses 1d ADX(14) > 20 to confirm trending regime, avoiding false breakouts in choppy markets. Works in bull/bear by following EMA trend direction.
"""

name = "4h_Donchian_Breakout_Volume_Trend"
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

    # Get 4h data for Donchian channels (same timeframe, but using rolling window)
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values

    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])

    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])

    # Smoothed TR, DM+
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)

    # Wilder's smoothing
    atr[tr_period] = np.nansum(tr[1:tr_period+1])
    dm_plus_smooth[tr_period] = np.nansum(dm_plus[1:tr_period+1])
    dm_minus_smooth[tr_period] = np.nansum(dm_minus[1:tr_period+1])

    for i in range(tr_period+1, len(tr)):
        atr[i] = atr[i-1] - (atr[i-1]/tr_period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1]/tr_period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1]/tr_period) + dm_minus[i]

    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]

    # DX and ADX
    dx = np.full_like(tr, np.nan)
    di_sum = di_plus + di_minus
    valid_dx = ~np.isnan(di_sum) & (di_sum != 0)
    dx[valid_dx] = 100 * np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]

    adx = np.full_like(tr, np.nan)
    adx[tr_period*2:] = pd.Series(dx[tr_period*2:]).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_1d = adx  # already aligned to 1d

    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    # 4h EMA(50) trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume spike: >1.5x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian upper + EMA50 uptrend + ADX > 20 + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50[i] and 
                adx_1d_aligned[i] > 20 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + EMA50 downtrend + ADX > 20 + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50[i] and 
                  adx_1d_aligned[i] > 20 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (reversal)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (reversal)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals