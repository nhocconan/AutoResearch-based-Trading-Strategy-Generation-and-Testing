#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d ADX Regime + Volume Confirmation
Hypothesis: Elder Ray (Bull/Bear Power) measures trend strength via EMA13 deviation.
In strong trends (ADX>25 on 1d), trade in direction of Power (Bull Power>0 long, Bear Power<0 short).
In ranging markets (ADX<20), fade extreme Power readings (mean reversion).
Uses 6h timeframe with 1d HTF for regime/ADX and EMA13 for Power calculation.
Targets 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
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
    
    # Get 1d data for ADX regime and EMA13 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close for Elder Ray Power
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(
        span=13, adjust=False, min_periods=13
    ).mean().values
    
    # Calculate ADX on 1d for regime filtering
    # ADX requires +DI, -DI, DX calculation
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full_like(high_arr, np.nan), np.full_like(high_arr, np.nan), np.full_like(high_arr, np.nan)
        
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # +DM and -DM
        up_move = high_arr[1:] - high_arr[:-1]
        down_move = low_arr[:-1] - low_arr[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        def WilderSmoothing(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            result = np.full_like(arr, np.nan)
            result[period-1] = np.nanmean(arr[:period])
            alpha = 1.0 / period
            for i in range(period, len(arr)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = WilderSmoothing(tr, period)
        plus_dm_smooth = WilderSmoothing(plus_dm, period)
        minus_dm_smooth = WilderSmoothing(minus_dm, period)
        
        # +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = WilderSmoothing(dx, period)
        
        return adx, plus_di, minus_di
    
    adx_1d, plus_di_1d, minus_di_1d = calculate_adx(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Calculate Elder Ray Power on 6s using 1d EMA13
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align all 1d indicators to 6h
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 20-period volume MA for 6h volume confirmation
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 14+14)  # 20 for volume MA, 28 for ADX (14+14 for Wilder smoothing)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema13 = ema_13_1d_aligned[i]
        adx = adx_1d_aligned[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Volume confirmation: current 6h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_6h
        
        # Regime classification
        trending = adx > 25
        ranging = adx < 20
        
        if position == 0:
            # Look for entry signals
            if trending and volume_confirm:
                # In trending regime: trade with the power
                long_entry = bull_power > 0
                short_entry = bear_power < 0
            elif ranging and volume_confirm:
                # In ranging regime: fade extreme power (mean reversion)
                long_entry = bear_power < -0.5 * curr_close * 0.01  # Bear power significantly negative
                short_entry = bull_power > 0.5 * curr_close * 0.01   # Bull power significantly positive
            else:
                long_entry = False
                short_entry = False
            
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
            # Exit: power turns negative OR ADX weakens (<20) OR volume drops
            if bull_power <= 0 or adx < 20 or curr_volume < vol_ma_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: power turns positive OR ADX weakens (<20) OR volume drops
            if bear_power >= 0 or adx < 20 or curr_volume < vol_ma_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_ADXRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0