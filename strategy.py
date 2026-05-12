#!/usr/bin/env python3
# 4h_Donchian_Breakout_VolumeTrend_Filter_v3
# Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter.
# Long: Close breaks above Donchian high(20) + volume > 1.5x SMA20 + price > daily EMA50
# Short: Close breaks below Donchian low(20) + volume > 1.5x SMA20 + price < daily EMA50
# Exit: Close crosses opposite Donchian level (low for long, high for short)
# Uses 4h timeframe for optimal trade frequency (target 20-50/year) to minimize fee drift.
# Includes volatility filter: only trade when 4h ATR < 1.5x 1d ATR to avoid choppy periods.

name = "4h_Donchian_Breakout_VolumeTrend_Filter_v3"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    # Volatility filter: 4h ATR < 1.5x 1d ATR
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = 0
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    tr1d = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_d = np.maximum(tr1d, tr2d)
    tr_d[0] = 0
    atr_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema50_aligned = ema50_1d_aligned[i]
        vol_threshold_val = volume_threshold[i]
        atr_4h_val = atr_4h[i]
        atr_1d_aligned_val = atr_1d_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or 
            np.isnan(ema50_aligned) or np.isnan(vol_threshold_val) or
            np.isnan(atr_4h_val) or np.isnan(atr_1d_aligned_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volatility filter: only trade when 4h ATR < 1.5x 1d ATR (avoid choppy periods)
        volatility_filter = atr_4h_val < 1.5 * atr_1d_aligned_val

        if position == 0:
            # LONG: Close breaks above Donchian high + volume spike + daily uptrend + volatility filter
            if (close[i] > donchian_high_val and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned and
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Donchian low + volume spike + daily downtrend + volatility filter
            elif (close[i] < donchian_low_val and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned and
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below Donchian low
            if close[i] < donchian_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above Donchian high
            if close[i] > donchian_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals