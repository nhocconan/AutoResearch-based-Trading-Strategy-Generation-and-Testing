#!/usr/bin/env python3
"""
6h Elder Ray Power + ADX Regime + Volume Spike
Hypothesis: Elder Ray (Bull/Bear Power) measures trend strength relative to EMA13. 
Combined with ADX regime filter (ADX>25 for trending, <20 for ranging) and volume spike confirmation,
it captures strong trend continuations while avoiding chop. Works in both bull/bear markets by 
trading with the trend when ADX confirms strength. Uses 6h timeframe with 1d HTF for EMA/ADX.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(
        span=13, adjust=False, min_periods=13
    ).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate ADX on 1d (14-period)
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full_like(close_arr, np.nan)
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high_arr[1:] - high_arr[:-1]
        down_move = low_arr[:-1] - low_arr[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        def wilder_smooth(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            result = np.full_like(arr, np.nan)
            result[period-1] = np.nanmean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_14_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate 20-period volume MA for 6h volume confirmation
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 13, 14*2)  # 20 for volume MA, 13 for EMA, 28 for ADX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema13 = ema_13_1d_aligned[i]
        adx = adx_14_1d_aligned[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Elder Ray Power
        bull_power = curr_high - ema13  # Bull Power: High - EMA13
        bear_power = ema13 - curr_low   # Bear Power: EMA13 - Low
        
        # Volume confirmation: current 6h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_6h
        
        # ADX regime: ADX > 25 = strong trend, ADX < 20 = ranging/chop
        strong_trend = adx > 25
        ranging = adx < 20
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0 AND Bear Power < Bull Power (bulls stronger) AND strong trend AND volume confirmation
            long_entry = (bull_power > 0 and bear_power < bull_power and 
                         strong_trend and volume_confirm)
            # Short: Bear Power > 0 AND Bull Power < Bear Power (bears stronger) AND strong trend AND volume confirmation
            short_entry = (bear_power > 0 and bull_power < bear_power and 
                          strong_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Bull Power <= 0 (weakness) OR ADX drops below 20 (trend ending) OR Bear Power > Bull Power (bears take over)
            if (bull_power <= 0 or adx < 20 or bear_power > bull_power):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bear Power <= 0 (weakness) OR ADX drops below 20 (trend ending) OR Bull Power > Bear Power (bulls take over)
            if (bear_power <= 0 or adx < 20 or bull_power > bear_power):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_ADXRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0