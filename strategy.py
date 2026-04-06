#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with daily volume confirmation and weekly ADX trend filter.
# Works in bull: breakouts capture momentum. Works in bear: ADX filter avoids whipsaws in low volatility,
# volume confirmation ensures only strong breakouts trigger entries. Target: 80-160 trades over 4 years.

name = "exp_12975_6h_donchian20_1d_vol_1w_adx_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily and weekly data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate daily indicators
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    volume_d = df_daily['volume'].values
    
    upper_d, lower_d = calculate_donchian(high_d, low_d, DONCHIAN_PERIOD)
    volume_ma_d = pd.Series(volume_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    adx_d = calculate_adx(high_d, low_d, close_d, ADX_PERIOD)
    
    # Align daily indicators to 6h
    upper_d_aligned = align_htf_to_ltf(prices, df_daily, upper_d)
    lower_d_aligned = align_htf_to_ltf(prices, df_daily, lower_d)
    volume_ma_d_aligned = align_htf_to_ltf(prices, df_daily, volume_ma_d)
    adx_d_aligned = align_htf_to_ltf(prices, df_daily, adx_d)
    
    # Calculate weekly indicators for trend filter
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    adx_w = calculate_adx(high_w, low_w, close_w, ADX_PERIOD)
    adx_w_aligned = align_htf_to_ltf(prices, df_weekly, adx_w)
    
    # Calculate 6h ATR for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    atr_6h = calculate_atr(high_6h, low_6h, close_6h, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(upper_d_aligned[i]) or np.isnan(lower_d_aligned[i]) or 
            np.isnan(volume_ma_d_aligned[i]) or np.isnan(adx_d_aligned[i]) or 
            np.isnan(adx_w_aligned[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close_6h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_6h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (daily)
        volume_ok = volume_6h := np.mean(volume_d[max(0, i-4):i+1]) if i >= 4 else volume_d[i]
        volume_ok = volume_6h > (volume_ma_d_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_d_aligned[i]) else False
        
        # ADX trend filter (both daily and weekly must agree)
        adx_ok = adx_d_aligned[i] > ADX_THRESHOLD and adx_w_aligned[i] > ADX_THRESHOLD
        
        # Breakout conditions
        breakout_long = volume_ok and adx_ok and close_6h[i] >= upper_d_aligned[i]
        breakout_short = volume_ok and adx_ok and close_6h[i] <= lower_d_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_6h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_6h[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_6h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_6h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals