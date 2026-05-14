#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h Supertrend trend filter and 1d volume spike confirmation.
# Uses Supertrend for robust trend detection (works in both bull and bear markets by adapting to volatility).
# Long when price breaks above R3 AND Supertrend gives bullish signal AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below S3 AND Supertrend gives bearish signal AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the prior day's close (Camarilla pivot point).
# Uses discrete position sizing (0.20) to limit fee churn. Target: 60-150 total trades over 4 years (15-37/year) for 1h.

name = "1h_Camarilla_R3S3_Breakout_4hSupertrend_Trend_1dVolumeConfirm_v1"
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
    
    # Calculate 4h Supertrend for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.full_like(close_4h, np.nan)
    direction = np.full_like(close_4h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    start_idx = atr_period
    if start_idx < len(close_4h):
        supertrend[start_idx] = upper_band[start_idx]
        direction[start_idx] = 1  # start with uptrend assumption
    
    # Calculate Supertrend iteratively
    for i in range(start_idx + 1, len(close_4h)):
        if close_4h[i-1] > supertrend[i-1]:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
        
        # Update direction
        if close_4h[i] > supertrend[i]:
            direction[i] = 1
        else:
            direction[i] = -1
    
    # Align Supertrend direction to 1h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, direction.astype(float))
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
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
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_cp[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND Supertrend bullish (direction=1) AND volume confirmation
            if (open_[i] <= camarilla_r3[i] and close[i] > camarilla_r3[i] and 
                supertrend_direction_aligned[i] > 0.5 and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S3 AND Supertrend bearish (direction=-1) AND volume confirmation
            elif (open_[i] >= camarilla_s3[i] and close[i] < camarilla_s3[i] and 
                  supertrend_direction_aligned[i] < -0.5 and 
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