#!/usr/bin/env python3
"""
12h_PowerTrend_VolumeConfirmation
12h strategy combining trend strength (ADX) with volume spikes and momentum.
- Long: ADX > 25, +DI > -DI, volume > 2x average, RSI > 50
- Short: ADX > 25, -DI > +DI, volume > 2x average, RSI < 50
- Exit: Opposite signal or ADX < 20 (trend weakening)
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in trending markets (both bull and bear) by capturing strong directional moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX and RSI calculations
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[1:period])
            # Subsequent values
            for i in range(period, len(values)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di_14 = np.where(tr_14 != 0, 100 * plus_dm_14 / tr_14, 0)
    minus_di_14 = np.where(tr_14 != 0, 100 * minus_dm_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    adx = wilders_smoothing(dx, 14)
    
    # RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = wilders_smoothing(np.concatenate([[np.nan], gain]), 14)
    avg_loss = wilders_smoothing(np.concatenate([[np.nan], loss]), 14)
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with original index
    
    # Volume average (20-period)
    vol_ma_20 = np.concatenate([[np.nan] * 19, 
                                np.convolve(volume_1d, np.ones(20)/20, mode='valid')])
    
    # Align all indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di_14)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di_14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend strength and direction
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        bullish = plus_di_aligned[i] > minus_di_aligned[i]
        bearish = minus_di_aligned[i] > plus_di_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma_aligned[i]
        
        # Momentum filter
        bullish_momentum = rsi_aligned[i] > 50
        bearish_momentum = rsi_aligned[i] < 50
        
        if position == 0:
            # Long: strong uptrend + volume + bullish momentum
            if strong_trend and bullish and vol_confirm and bullish_momentum:
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend + volume + bearish momentum
            elif strong_trend and bearish and vol_confirm and bearish_momentum:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakening or bearish reversal
            if weak_trend or (bearish and vol_confirm and bearish_momentum):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakening or bullish reversal
            if weak_trend or (bullish and vol_confirm and bullish_momentum):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PowerTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0