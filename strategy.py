#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover (8/21) with 4h ADX trend filter and 1d volume regime filter.
Long when 1h EMA8 crosses above EMA21, 4h ADX > 25 (trending), and 1d volume > 1.5x 20-day average.
Short when 1h EMA8 crosses below EMA21, 4h ADX > 25, and 1d volume > 1.5x 20-day average.
Exit when EMA crossover reverses or ADX < 20 (range market).
Uses 4h for trend strength (ADX), 1d for volume participation, 1h for precise entry timing.
Designed to capture strong trending moves with volume confirmation in both bull and bear markets.
Target: 15-35 trades/year per symbol to minimize fee drag on 1h timeframe.
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
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h ADX (14-period)
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # first value is simple average
            result[period-1] = np.nanmean(data[:period])
            # subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_4h = wilders_smoothing(tr, 14)
    plus_dm_4h = wilders_smoothing(plus_dm, 14)
    minus_dm_4h = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di_4h = np.where(atr_4h != 0, (plus_dm_4h / atr_4h) * 100, 0)
    minus_di_4h = np.where(atr_4h != 0, (minus_dm_4h / atr_4h) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di_4h + minus_di_4h) != 0, 
                  np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h) * 100, 0)
    adx_4h = wilders_smoothing(dx, 14)
    
    # Calculate 1h EMA8 and EMA21 for entry signals
    close_series = pd.Series(close)
    ema8_1h = close_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_1h = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d volume MA20 for regime filter
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 4h ADX and 1d volume MA to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA21 and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema8_1h[i]) or 
            np.isnan(ema21_1h[i]) or 
            np.isnan(adx_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 1d volume > 1.5x 20-day average (expanding participation)
        # Get the aligned 1d volume for this timestamp (we need to align volume_1d as well)
        df_1d_full = get_htf_data(prices, '1d')  # reload to get volume column
        volume_1d_full = df_1d_full['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_full)
        volume_confirmed = not np.isnan(volume_1d_aligned[i]) and \
                          not np.isnan(vol_ma_20_1d_aligned[i]) and \
                          volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # EMA crossover signals
        ema8_prev = ema8_1h[i-1] if i > 0 else np.nan
        ema21_prev = ema21_1h[i-1] if i > 0 else np.nan
        ema8_cross_above = ema8_prev <= ema21_prev and ema8_1h[i] > ema21_1h[i]
        ema8_cross_below = ema8_prev >= ema21_prev and ema8_1h[i] < ema21_1h[i]
        
        if position == 0:
            # Long: EMA8 crosses above EMA21, strong trend (ADX > 25), volume confirmation
            if (ema8_cross_above and 
                adx_4h_aligned[i] > 25 and 
                volume_confirmed):
                signals[i] = 0.20
                position = 1
            # Short: EMA8 crosses below EMA21, strong trend (ADX > 25), volume confirmation
            elif (ema8_cross_below and 
                  adx_4h_aligned[i] > 25 and 
                  volume_confirmed):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: EMA8 crosses below EMA21 OR trend weakens (ADX < 20)
            if (ema8_cross_below or adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: EMA8 crosses above EMA21 OR trend weakens (ADX < 20)
            if (ema8_cross_above or adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA8_21_4hADX25_1dVolumeRegime"
timeframe = "1h"
leverage = 1.0