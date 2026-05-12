#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Hypothesis: Use 4h trend and 1d volume regime to filter 1h Camarilla R1/S1 breakouts.
In trending markets (4h EMA50 up/down) with elevated volume (1d volume > 1.5x 20-day average),
buy breaks above R1, sell breaks below S1. This avoids low-volume false breakouts and
whipsaws in ranging markets. Targets 15-35 trades/year by combining multiple filters.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
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
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values

    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values

    # Calculate Camarilla levels from previous day
    # Using previous 1d OHLC (already available in df_1d)
    # Roll to get previous day's values
    high_1d_prev = np.roll(df_1d['high'].values, 1)
    low_1d_prev = np.roll(df_1d['low'].values, 1)
    close_1d_prev = np.roll(df_1d['close'].values, 1)
    # First values invalid
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan

    # Camarilla R1 and S1 calculation
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 12
    camarilla_s1 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 12

    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # 1d volume regime: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume_1d > (vol_ma_20 * 1.5)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 4h uptrend + high volume regime
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_4h_aligned[i] and 
                vol_regime_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + 4h downtrend + high volume regime
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  vol_regime_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals