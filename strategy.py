#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d ADX Regime + Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
Combine with 1d ADX regime filter: ADX > 25 = trending (trade Elder Ray extremes), ADX < 20 = ranging (fade extremes).
Volume confirmation (>1.5x 20-bar vol MA) ensures institutional participation.
Works in bull markets via strong Bull Power + uptrend regime, and in bear markets via strong Bear Power + downtrend regime.
Discrete sizing (0.25) targets 50-150 total trades over 4 years to minimize fee drag.
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
    
    # Get 1d data for ADX regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    plus_dm_1d = np.zeros(len(df_1d))
    minus_dm_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm_1d[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_1d[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(values[1:period])
        # Wilder smoothing: result[i] = (result[i-1] * (period-1) + values[i]) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    tr14_1d = wilders_smooth(tr_1d, 14)
    plus_dm14_1d = wilders_smooth(plus_dm_1d, 14)
    minus_dm14_1d = wilders_smooth(minus_dm_1d, 14)
    
    # Avoid division by zero
    plus_di14_1d = np.where(tr14_1d != 0, (plus_dm14_1d / tr14_1d) * 100, 0)
    minus_di14_1d = np.where(tr14_1d != 0, (minus_dm14_1d / tr14_1d) * 100, 0)
    
    dx_1d = np.where((plus_di14_1d + minus_di14_1d) != 0, 
                     (abs(plus_di14_1d - minus_di14_1d) / (plus_di14_1d + minus_di14_1d)) * 100, 0)
    adx_1d = wilders_smooth(dx_1d, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Calculate 20-period volume MA for volume confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13, ADX, volume MA to propagate
    start_idx = max(13, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema13_val = ema_13[i]
        adx_val = adx_1d_aligned[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # Regime filters
        trending_market = adx_val > 25
        ranging_market = adx_val < 20
        
        if position == 0:
            # Long entry: strong Bull Power + trending market (ADX>25) + volume confirmation
            long_entry = (bull_power_val > 0) and trending_market and volume_confirm
            # Short entry: strong Bear Power + trending market (ADX>25) + volume confirmation
            short_entry = (bear_power_val > 0) and trending_market and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power turns positive OR ADX drops below 20 (trend weakening)
            if bear_power_val > 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power turns positive OR ADX drops below 20 (trend weakening)
            if bull_power_val > 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0