#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with daily ADX filter and volume confirmation.
# Works in bull/bear by capturing strong trends (ADX>25) with volume confirmation,
# while avoiding whipsaws in ranging markets. Weekly pivot direction provides
# structural bias to reduce false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_12887_6h_donchian20_1d_adx_vol_wp_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
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
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily ADX
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    adx_d = calculate_adx(high_d, low_d, close_d, ADX_PERIOD)
    adx_d_aligned = align_htf_to_ltf(prices, df_daily, adx_d)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot_vals = np.zeros(len(close_w))
    r1_vals = np.zeros(len(close_w))
    r2_vals = np.zeros(len(close_w))
    r3_vals = np.zeros(len(close_w))
    s1_vals = np.zeros(len(close_w))
    s2_vals = np.zeros(len(close_w))
    s3_vals = np.zeros(len(close_w))
    
    for i in range(len(close_w)):
        pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(high_w[i], low_w[i], close_w[i])
        pivot_vals[i] = pivot
        r1_vals[i] = r1
        r2_vals[i] = r2
        r3_vals[i] = r3
        s1_vals[i] = s1
        s2_vals[i] = s2
        s3_vals[i] = s3
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3_vals)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3_vals)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donch_up, donch_low = calculate_donchian(high, low, DONCHIAN_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_d_aligned[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # ADX trend filter
        trend_strong = adx_d_aligned[i] > ADX_THRESHOLD
        
        # Weekly pivot bias
        pivot_bias_up = close[i] > pivot_aligned[i]
        pivot_bias_down = close[i] < pivot_aligned[i]
        
        # Breakout conditions
        breakout_long = volume_ok and trend_strong and pivot_bias_up and close[i] >= donch_up[i]
        breakout_short = volume_ok and trend_strong and pivot_bias_down and close[i] <= donch_low[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals