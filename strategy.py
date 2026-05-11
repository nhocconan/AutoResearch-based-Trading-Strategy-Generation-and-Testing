#!/usr/bin/env python3
name = "6h_ADX_Alligator_Trend"
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
    
    # Calculate Williams Alligator (SMMA-based)
    jaw_len = 13
    teeth_len = 8
    lips_len = 5
    jaw_offset = 8
    teeth_offset = 5
    lips_offset = 3
    
    # SMMA calculation (smoothed moving average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, jaw_len)  # Blue line
    teeth = smma(low, teeth_len)  # Red line
    lips = smma(close, lips_len)  # Green line
    
    # Apply offsets (shift right into future, so we need to lag them)
    jaw = np.roll(jaw, jaw_offset)
    teeth = np.roll(teeth, teeth_offset)
    lips = np.roll(lips, lips_offset)
    
    # Calculate ADX
    adx_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First TR has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    def smooth_rma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Wilder's smoothing: prev_value * (period-1)/period + current_value/period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_smoothed = smooth_rma(tr, adx_period)
    plus_dm_smoothed = smooth_rma(plus_dm, adx_period)
    minus_dm_smoothed = smooth_rma(minus_dm, adx_period)
    
    # Directional indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_rma(dx, adx_period)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # ADX trend strength filter (trending when ADX > 25)
        strong_trend = adx[i] > 25
        
        # 1d EMA50 trend filter
        trend_up_1d = close[i] > ema_50_1d_aligned[i]
        trend_down_1d = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up + strong trend + price above 1d EMA50 + volume
            if alligator_long and strong_trend and trend_up_1d and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + strong trend + price below 1d EMA50 + volume
            elif alligator_short and strong_trend and trend_down_1d and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator breaks down OR ADX weakens (<20) OR price crosses below 1d EMA50
            if not alligator_long or adx[i] < 20 or not trend_up_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator breaks up OR ADX weakens (<20) OR price crosses above 1d EMA50
            if not alligator_short or adx[i] < 20 or not trend_down_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals