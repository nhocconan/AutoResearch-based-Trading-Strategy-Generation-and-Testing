#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 indicates strong trend (use Elder Ray signals in trend direction)
# 1d ADX < 20 indicates ranging market (fade extreme Elder Ray readings)
# Volume confirmation filters low-conviction moves
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets via bull power longs in uptrends and bear power shorts in downtrends
# Works in bear/ranging markets via mean reversion at extreme Elder Ray levels

name = "6h_ElderRay_1dADXRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = pd.Series(df_1d['close'].values)
    ema_13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX (14-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d_series = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d_series.shift(1))
    tr3 = abs(low_1d - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d.diff()
    down_move = low_1d.diff().multiply(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema_13_6h = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA and aligned HTF
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13_6h[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Regime-based entries
            if adx_1d_aligned[i] > 25:  # Trending market
                # Long: strong bull power in uptrend
                if bull_power[i] > 0 and volume_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: strong bear power in downtrend
                elif bear_power[i] < 0 and volume_spike:
                    signals[i] = -0.25
                    position = -1
            elif adx_1d_aligned[i] < 20:  # Ranging market
                # Long: extreme bear power (oversold) with volume spike
                if bear_power[i] < (-2 * np.std(bull_power[max(0, i-50):i])) and volume_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: extreme bull power (overbought) with volume spike
                elif bull_power[i] > (2 * np.std(bear_power[max(0, i-50):i])) and volume_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: bear power turns negative or loses volume confirmation
            if bear_power[i] < 0 or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power turns positive or loses volume confirmation
            if bull_power[i] > 0 or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals