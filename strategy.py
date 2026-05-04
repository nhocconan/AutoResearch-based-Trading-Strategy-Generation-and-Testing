#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Uses 1d ADX(14) > 25 to identify strong trends (reduces whipsaw in ranging markets)
# Donchian breakout captures momentum in trending regimes
# Volume confirmation (>1.8x 20 EMA) ensures institutional participation
# Session filter (08-20 UTC) focuses on high-liquidity periods
# Discrete sizing 0.25 manages risk and minimizes fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# Works in both bull and bear: ADX filter ensures we only trade strong trends.

name = "4h_Donchian20_1dADX_VolumeConfirm_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilder_smooth(values):
        smoothed = np.full_like(values, np.nan, dtype=float)
        for i in range(len(values)):
            if np.isnan(values[i]):
                if i == 0:
                    smoothed[i] = np.nan
                else:
                    smoothed[i] = smoothed[i-1]
            else:
                if i == 0 or np.isnan(smoothed[i-1]):
                    smoothed[i] = values[i]
                else:
                    smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    tr_smoothed = wilder_smooth(tr)
    plus_dm_smoothed = wilder_smooth(plus_dm)
    minus_dm_smoothed = wilder_smooth(minus_dm)
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx)
    
    # Align 1d ADX to 4h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 4h timeframe
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + strong trend (ADX>25) + volume spike
            if close[i] > upper_channel[i] and adx_aligned[i] > 25 and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + strong trend (ADX>25) + volume spike
            elif close[i] < lower_channel[i] and adx_aligned[i] > 25 and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to mid-channel OR trend weakens (ADX<20) OR weak volume
            mid_channel = (upper_channel[i] + lower_channel[i]) / 2.0
            if (close[i] < mid_channel or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to mid-channel OR trend weakens (ADX<20) OR weak volume
            mid_channel = (upper_channel[i] + lower_channel[i]) / 2.0
            if (close[i] > mid_channel or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals