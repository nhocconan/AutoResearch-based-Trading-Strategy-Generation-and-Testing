#!/usr/bin/env python3
"""
6h Elder Ray Index + 1d ADX Trend Filter + Volume Spike Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength relative to trend.
In strong trends (ADX > 25 on 1d), we take trades in direction of Elder Ray with volume confirmation.
In weak trends (ADX <= 25), we stay flat to avoid whipsaw.
Uses 6h primary timeframe with 1d ADX for trend strength and 1d EMA13 for Elder Ray calculation.
Designed for BTC/ETH with 50-150 total trades over 4 years to minimize fee drag while capturing strong trends.
"""

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
    
    # Get 1d data for ADX and EMA13 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX and EMA13
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = pd.Series(df_1d['close'])
    ema_13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX (14-period)
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d_series = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d_series.shift(1))
    tr3 = abs(low_1d - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d.diff()
    down_move = low_1d.diff() * -1  # inverted so down move is positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13, ADX, and volume MA
    start_idx = max(30, 20)  # 30 for ADX/EMA13, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_13_val = ema_13_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Elder Ray calculations
        bull_power = curr_high - ema_13_val
        bear_power = curr_low - ema_13_val
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if not strong_trend:
                # Weak trend: stay flat to avoid whipsaw
                signals[i] = 0.0
                position = 0
            elif bull_power > 0 and bear_power < 0:
                # Both bull and bear power present - wait for clearer signal
                signals[i] = 0.0
                position = 0
            elif bull_power > 0:
                # Bull power dominant: look for long with volume confirmation
                long_signal = volume_confirm
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
                    position = 0
            elif bear_power < 0:
                # Bear power dominant: look for short with volume confirmation
                short_signal = volume_confirm
                if short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
        elif position == 1:
            # Exit long: bear power becomes dominant OR ADX weakens
            if bear_power >= 0 or adx_val <= 20:  # hysteresis: exit at 20 to avoid chattering
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power becomes dominant OR ADX weakens
            if bull_power <= 0 or adx_val <= 20:  # hysteresis: exit at 20 to avoid chattering
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0