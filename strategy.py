#!/usr/bin/env python3
# Hypothesis: 4h Williams %R extreme reversal with 1d ADX regime filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; ADX > 25 filters for trending markets
# to avoid false reversals in ranging markets; volume spike confirms conviction. Discrete sizing
# (0.0, ±0.30) minimizes fee churn. Designed to work in both bull (buy oversold in uptrend) and
# bear (sell overbought in downtrend) markets by trading reversals only when trend strength
# confirms momentum continuation after pullback.

name = "4h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14 period): (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 = overbought, < -80 = oversold
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100
    # Avoid division by zero
    denom = highest_high_14 - lowest_low_14
    denom = np.where(denom == 0, 1e-10, denom)
    williams_r = ((highest_high_14 - close) / denom) * -100
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX (14 period) - measures trend strength
    # +DM = high[t] - high[t-1] (if positive and > low[t-1] - low[t])
    # -DM = low[t-1] - low[t] (if positive and > high[t] - high[t-1])
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    # +DI = 100 * EWMA(+DM) / ATR, -DI = 100 * EWMA(-DM) / ATR
    # ADX = 100 * EWMA(|+DI - -DI| / (+DI + -DI))
    
    # Calculate +DM and -DM
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initial TR, +DM, -DM sums
    tr_sum = np.zeros_like(tr)
    plus_dm_sum = np.zeros_like(plus_dm)
    minus_dm_sum = np.zeros_like(minus_dm)
    
    # First value is simple average
    tr_sum[period-1] = tr[:period].sum()
    plus_dm_sum[period-1] = plus_dm[:period].sum()
    minus_dm_sum[period-1] = minus_dm[:period].sum()
    
    # Wilder smoothing: today = (yesterday * (period-1) + today) / period
    for i in range(period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # Calculate DX and ADX
    dx = np.zeros_like(tr)
    dx_mask = (plus_di + minus_di) > 0
    dx[dx_mask] = 100 * np.abs(plus_di[dx_mask] - minus_di[dx_mask]) / (plus_di[dx_mask] + minus_di[dx_mask])
    
    # ADX is Wilder smoothed DX
    adx = np.zeros_like(dx)
    adx[period-1] = dx[:period].mean()  # First ADX is average of first period DX
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX to 4h (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R lookback
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when trending (ADX > 25)
        if adx_aligned[i] <= 25:
            # In ranging market, stay flat to avoid false reversals
            signals[i] = 0.0
            continue
        
        # Trading logic: Williams %R extremes in trending market
        if position == 0:
            # LONG: Oversold (< -80) AND volume spike
            if williams_r[i] < -80.0 and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Overbought (> -20) AND volume spike
            elif williams_r[i] > -20.0 and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 (momentum fading) OR overbought
            if williams_r[i] > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 (momentum fading) OR oversold
            if williams_r[i] < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals