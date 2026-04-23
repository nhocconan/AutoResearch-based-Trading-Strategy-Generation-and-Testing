#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d Regime Filter
- Uses Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) from 6h
- 1d ADX < 20 defines ranging market (mean reversion regime)
- In ranging markets: fade extreme Elder Ray readings (Bull Power < -std AND Bear Power > std for short, vice versa for long)
- In trending markets (ADX >= 20): follow Elder Ray momentum (Bull Power > 0 for long, Bear Power < 0 for short)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum signals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by adapting to regime (trend vs range)
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
    
    # Calculate 6h Elder Ray components
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # EMA13 for Elder Ray
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema13_6h = ema(close_6h, 13)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_6h = high_6h - ema13_6h
    bear_power_6h = low_6h - ema13_6h
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # Calculate 1d ADX regime filter (ADX < 20 = ranging, ADX >= 20 = trending)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # Calculate Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di_14 = np.where(atr_1d != 0, (plus_dm_14 / atr_1d) * 100, 0)
    minus_di_14 = np.where(atr_1d != 0, (minus_dm_14 / atr_1d) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Elder Ray volatility (for extreme readings in ranging markets)
    def rolling_std(values, window):
        return pd.Series(values).rolling(window=window, min_periods=window).std().values
    
    bull_power_std = rolling_std(bull_power_aligned, 20)
    bear_power_std = rolling_std(bear_power_aligned, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30, 20)  # Elder Ray, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(bull_power_std[i]) or np.isnan(bear_power_std[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime
        ranging_market = adx_aligned[i] < 20
        trending_market = adx_aligned[i] >= 20
        
        if position == 0:
            if ranging_market:
                # Ranging market: mean reversion at extreme Elder Ray readings
                long_signal = (bear_power_aligned[i] < -bull_power_std[i] and  # Extreme bear power
                              volume[i] > 1.5 * vol_ma[i])
                short_signal = (bull_power_aligned[i] > bear_power_std[i] and   # Extreme bull power
                               volume[i] > 1.5 * vol_ma[i])
            else:  # trending_market
                # Trending market: follow Elder Ray momentum
                long_signal = (bull_power_aligned[i] > 0 and  # Positive bull power
                              volume[i] > 1.5 * vol_ma[i])
                short_signal = (bear_power_aligned[i] < 0 and  # Negative bear power
                               volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: in ranging market when bear power normalizes OR in trending when bull power turns negative
                if ranging_market and bear_power_aligned[i] > -0.5 * bull_power_std[i]:
                    exit_signal = True
                elif trending_market and bull_power_aligned[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: in ranging market when bull power normalizes OR in trending when bear power turns positive
                if ranging_market and bull_power_aligned[i] < 0.5 * bear_power_std[i]:
                    exit_signal = True
                elif trending_market and bear_power_aligned[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dADXRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0