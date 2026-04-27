#!/usr/bin/env python3
"""
4h ADX + Bollinger Band Width Regime Filter with 1-day EMA Trend.
Long when ADX > 25 (trending) + BBW < 50th percentile (low volatility) + price > 1-day EMA.
Short when ADX > 25 (trending) + BBW < 50th percentile (low volatility) + price < 1-day EMA.
Uses ADX for trend strength, BBW for volatility regime, and daily EMA for direction.
Designed for low frequency (20-40 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, length=14):
    """Average Directional Index (ADX)"""
    if len(high) < length + 1:
        return np.full_like(high, np.nan, dtype=np.float64)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM-
    atr = np.zeros_like(tr, dtype=np.float64)
    dm_plus_smooth = np.zeros_like(dm_plus, dtype=np.float64)
    dm_minus_smooth = np.zeros_like(dm_minus, dtype=np.float64)
    
    # Initial values (simple average)
    atr[length-1] = np.mean(tr[:length])
    dm_plus_smooth[length-1] = np.mean(dm_plus[:length])
    dm_minus_smooth[length-1] = np.mean(dm_minus[:length])
    
    # Wilder smoothing
    for i in range(length, len(tr)):
        atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (length-1) + dm_plus[i]) / length
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (length-1) + dm_minus[i]) / length
    
    # Directional Indicators
    plus_di = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(dx, dtype=np.float64)
    adx[2*length-2] = np.mean(dx[length-1:2*length-1])  # First ADX value
    
    for i in range(2*length-1, len(dx)):
        adx[i] = (adx[i-1] * (length-1) + dx[i]) / length
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day EMA34 for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])  # Simple average for first value
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 33) / 35  # EMA 34
    
    # Align 1-day EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ADX on 4h data (trend strength)
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate Bollinger Band Width on 4h data (volatility regime)
    bb_length = 20
    bb_std = 2.0
    sma = np.full_like(close, np.nan, dtype=np.float64)
    bb_width = np.full_like(close, np.nan, dtype=np.float64)
    
    if len(close) >= bb_length:
        # Calculate SMA
        for i in range(bb_length-1, len(close)):
            sma[i] = np.mean(close[i-bb_length+1:i+1])
        
        # Calculate standard deviation
        for i in range(bb_length-1, len(close)):
            bb_std_dev = np.std(close[i-bb_length+1:i+1])
            upper = sma[i] + bb_std * bb_std_dev
            lower = sma[i] - bb_std * bb_std_dev
            bb_width[i] = (upper - lower) / sma[i] * 100 if sma[i] != 0 else 0
    
    # Calculate percentile rank of BBW (20-period lookback)
    bb_width_percentile = np.full_like(bb_width, np.nan, dtype=np.float64)
    lookback = 20
    for i in range(lookback, len(bb_width)):
        if not np.isnan(bb_width[i]):
            # Calculate percentile: percentage of values in lookback window that are <= current value
            window = bb_width[i-lookback:i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                percentile = (np.sum(valid_window <= bb_width[i]) / len(valid_window)) * 100
                bb_width_percentile[i] = percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ADX (14*2=28), BBW (20+20=40), EMA (34)
    start_idx = max(40, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and indicators
        price_now = close[i]
        adx_now = adx[i]
        bbw_percentile = bb_width_percentile[i]
        ema_trend = ema_34_aligned[i]
        
        # Regime filters
        trending = adx_now > 25              # Strong trend
        low_volatility = bbw_percentile < 50 # Low volatility regime
        
        if position == 0:
            # Long: trending + low volatility + price above daily EMA
            if trending and low_volatility and price_now > ema_trend:
                signals[i] = size
                position = 1
            # Short: trending + low volatility + price below daily EMA
            elif trending and low_volatility and price_now < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakens OR volatility increases OR price crosses below EMA
            if (adx_now < 20) or (bbw_percentile > 70) or (price_now < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend weakens OR volatility increases OR price crosses above EMA
            if (adx_now < 20) or (bbw_percentile > 70) or (price_now > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ADX_BBWRegime_DailyEMA_Trend"
timeframe = "4h"
leverage = 1.0