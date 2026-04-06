#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 13-period EMA and trend filter using weekly ADX.
# Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Trend filter: weekly ADX > 25 indicates strong trend, enabling trend-following entries.
# Entry: Long when Bull Power > 0 and weekly ADX > 25; Short when Bear Power < 0 and weekly ADX > 25.
# Exit: Opposite signal or Elder Ray power crosses zero.
# Works in bull trends via bull power strength and bear trends via bear power strength.
# Target: 75-200 trades over 4 years (19-50/year).

name = "6h_elder_ray_trend_filter_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray: EMA(13) of close
    ema13 = np.full(n, np.nan)
    k = 2 / (13 + 1)
    for i in range(n):
        if i == 0:
            ema13[i] = close[i]
        elif np.isnan(ema13[i-1]):
            ema13[i] = close[i]
        else:
            ema13[i] = k * close[i] + (1 - k) * ema13[i-1]
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly ADX for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    n_w = len(close_1w)
    
    # True Range and Directional Movement
    tr = np.zeros(n_w)
    dm_plus = np.zeros(n_w)
    dm_minus = np.zeros(n_w)
    
    for i in range(1, n_w):
        tr0 = high_1w[i] - low_1w[i]
        tr1 = abs(high_1w[i] - close_1w[i-1])
        tr2 = abs(low_1w[i] - close_1w[i-1])
        tr[i] = max(tr0, tr1, tr2)
        
        dm_plus[i] = high_1w[i] - high_1w[i-1] if (high_1w[i] - high_1w[i-1]) > (low_1w[i-1] - low_1w[i]) and (high_1w[i] - high_1w[i-1]) > 0 else 0
        dm_minus[i] = low_1w[i-1] - low_1w[i] if (low_1w[i-1] - low_1w[i]) > (high_1w[i] - high_1w[i-1]) and (low_1w[i-1] - low_1w[i]) > 0 else 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilders_smoothing(arr, period):
        n = len(arr)
        result = np.full(n, np.nan)
        if n < period:
            return result
        # First value: average of first 'period' values
        result[period-1] = np.nanmean(arr[0:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1 / period
        for i in range(period, n):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] + alpha * (arr[i] - result[i-1])
        return result
    
    atr_w = wilders_smoothing(tr, 14)
    dm_plus_w = wilders_smoothing(dm_plus, 14)
    dm_minus_w = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full(n_w, np.nan)
    di_minus = np.full(n_w, np.nan)
    for i in range(14, n_w):
        if atr_w[i] > 0:
            di_plus[i] = 100 * dm_plus_w[i] / atr_w[i]
            di_minus[i] = 100 * dm_minus_w[i] / atr_w[i]
        else:
            di_plus[i] = 0
            di_minus[i] = 0
    
    # DX and ADX
    dx = np.full(n_w, np.nan)
    for i in range(14, n_w):
        if (di_plus[i] + di_minus[i]) > 0:
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        else:
            dx[i] = 0
    
    adx_w = wilders_smoothing(dx, 14)
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if weekly ADX not ready
        if np.isnan(adx_w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Strong trend filter: weekly ADX > 25
        strong_trend = adx_w_aligned[i] > 25
        
        if position == 0:
            # Look for new entries only in strong trend
            if strong_trend:
                # Long: Bull Power > 0
                if bull_power[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0
                elif bear_power[i] < 0:
                    signals[i] = -0.25
                    position = -1
        else:
            # Manage existing position
            if position == 1:
                # Exit long: Bull Power <= 0 or trend weakens
                if bull_power[i] <= 0 or not strong_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Bear Power >= 0 or trend weakens
                if bear_power[i] >= 0 or not strong_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals