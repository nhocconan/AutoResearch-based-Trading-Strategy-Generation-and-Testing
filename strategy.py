#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX trend filter and 1d volume spike confirmation.
# Long when Bull Power > 0 AND 1d ADX > 25 (strong trend) AND 1d volume > 2.0 * 20-period average volume.
# Short when Bear Power < 0 AND 1d ADX > 25 (strong trend) AND 1d volume > 2.0 * 20-period average volume.
# Exit when the respective power crosses zero (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit).
# Uses discrete position sizing (0.25) to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year) for 6h.
# Works in both bull and bear markets: 1d ADX filter ensures we only trade in clear trending conditions,
# while volume confirmation avoids breakouts in low-participation environments. Elder Ray adapts to volatility.

name = "6h_ElderRay_1dADXTrend_1dVolumeConfirm_v2"
timeframe = "6h"
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
    
    # Calculate 1d ADX trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, period)
    
    # Strong trend: ADX > 25
    strong_trend = adx > 25
    
    # Calculate 1d volume confirmation filter
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = df_1d['volume'].values > (2.0 * vol_ma_20_1d)
    
    # Align to 6h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (on 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(strong_trend_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND strong 1d trend AND volume confirmation
            if (bull_power[i] > 0 and 
                strong_trend_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND strong 1d trend AND volume confirmation
            elif (bear_power[i] < 0 and 
                  strong_trend_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals