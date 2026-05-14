#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX trend filter and 4h volume confirmation.
# Long when price breaks above R3 with 1d ADX > 25 (trending) and 4h volume > 1.8x 20-period average.
# Short when price breaks below S3 with 1d ADX > 25 (trending) and 4h volume > 1.8x 20-period average.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn. ADX > 25 ensures we only trade in trending markets,
# avoiding whipsaws in ranging conditions. Volume confirmation adds momentum validation.
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe.
# Works in bull/bear: 1d ADX filters for trend strength, Camarilla provides precise entry/exit levels.

name = "4h_Camarilla_R3S3_Breakout_1dADX_4hVolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 1.8x 20-period average (balanced filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) - measures trend strength
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM = high - prev_high (if positive and > low - prev_low)
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_plus[0] = 0
    # -DM = prev_low - low (if positive and > high - prev_high)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_minus[0] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA-like with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilder_smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # --- 4h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Precompute prior day's OHLC for each 4h bar using vectorized approach
    open_time = prices['open_time']
    prior_day_start = open_time - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
    
    # Create a mapping from date to prior day's OHLC
    pivot_dict = {}
    for _, row in df_1d_pivot.iterrows():
        day_start = row['open_time'].normalize()
        pivot_dict[day_start] = {
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
    
    for i in range(n):
        pd_ts = prior_day_start.iloc[i]
        if pd_ts in pivot_dict:
            data = pivot_dict[pd_ts]
            high_val = data['high']
            low_val = data['low']
            close_val = data['close']
            range_val = high_val - low_val
            camarilla_r3[i] = close_val + (range_val * 1.1 / 4)  # R3
            camarilla_s3[i] = close_val - (range_val * 1.1 / 4)  # S3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_confirm_4h[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + 1d ADX > 25 (trending) + 4h volume confirmation
            if (close[i] > camarilla_r3[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 1d ADX > 25 (trending) + 4h volume confirmation
            elif (close[i] < camarilla_s3[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals