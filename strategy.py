#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h ADX trend filter and 1d volume spike confirmation.
# Long when price breaks above R3 AND 4h ADX > 25 (strong trend) AND 1d volume > 1.8 * 20-period average volume.
# Short when price breaks below S3 AND 4h ADX > 25 (strong trend) AND 1d volume > 1.8 * 20-period average volume.
# Exit when price retraces to the prior day's close (Camarilla pivot point).
# Uses discrete position sizing (0.20) to limit fee churn. Designed for 1h timeframe with strict entry conditions.
# Session filter (08-20 UTC) to reduce noise trades. Target: 60-150 total trades over 4 years (15-37/year) for 1h.
# ADX filter ensures we only trade strong trends, reducing whipsaws in ranging markets.

name = "1h_Camarilla_R3S3_Breakout_4hADX25_1dVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h ADX for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_period = 14
    tr_smooth = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.full_like(tr_smooth, np.nan)
    di_minus = np.full_like(tr_smooth, np.nan)
    valid = ~np.isnan(tr_smooth) & (tr_smooth != 0)
    di_plus[valid] = (dm_plus_smooth[valid] / tr_smooth[valid]) * 100
    di_minus[valid] = (dm_minus_smooth[valid] / tr_smooth[valid]) * 100
    
    # DX and ADX
    dx = np.full_like(tr_smooth, np.nan)
    di_sum = di_plus + di_minus
    valid_dx = ~np.isnan(di_sum) & (di_sum != 0)
    dx[valid_dx] = (np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]) * 100
    
    adx = wilders_smoothing(dx, atr_period)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.8 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Camarilla pivot points (based on previous day's OHLC)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_cp = np.full(n, np.nan)  # Pivot point (close of prior day)
    
    # For each 1h bar, use prior completed day's OHLC
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        prior_day_start = current_time.normalize() - pd.Timedelta(days=1)
        prior_day_end = prior_day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        day_mask = (df_1d['open_time'] >= prior_day_start) & (df_1d['open_time'] <= prior_day_end)
        if day_mask.any():
            prior_day = df_1d.loc[day_mask].iloc[0]
            high_prior = prior_day['high']
            low_prior = prior_day['low']
            close_prior = prior_day['close']
            
            range_prior = high_prior - low_prior
            camarilla_r3[i] = close_prior + range_prior * 1.1 / 4  # R3 level
            camarilla_s3[i] = close_prior - range_prior * 1.1 / 4  # S3 level
            camarilla_cp[i] = close_prior  # Camarilla pivot point is the prior day's close
        else:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_cp[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute session hours to avoid datetime operations in loop
    session_hours = prices.index.hour
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_cp[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = session_hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND 4h ADX > 25 (strong trend) AND volume confirmation
            if (open_[i] <= camarilla_r3[i] and close[i] > camarilla_r3[i] and 
                adx_4h_aligned[i] > 25 and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S3 AND 4h ADX > 25 (strong trend) AND volume confirmation
            elif (open_[i] >= camarilla_s3[i] and close[i] < camarilla_s3[i] and 
                  adx_4h_aligned[i] > 25 and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Camarilla pivot point (CP)
            if close[i] <= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price retraces to Camarilla pivot point (CP)
            if close[i] >= camarilla_cp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals