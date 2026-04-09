#!/usr/bin/env python3
# 6h_elder_ray_regime_v3
# Hypothesis: 6h strategy using Elder Ray Index (Bull/Bear Power) with 1d ADX regime filter.
# In trending markets (ADX > 25), Elder Ray signals capture momentum continuations.
# In ranging markets (ADX < 20), fade extreme Elder Ray readings for mean reversion.
# Volume confirmation filters weak signals. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 80-160 total trades over 4 years by requiring regime alignment + Elder Ray extreme + volume.
# Primary timeframe: 6h, HTF: 1d for ADX regime and Elder Ray smoothing.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ADX regime and EMA smoothing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA22 for Elder Ray smoothing
    close_1d_series = pd.Series(close_1d)
    ema22_1d = close_1d_series.ewm(span=22, min_periods=22, adjust=False).mean().values
    
    # Calculate 1d ADX for regime detection (14-period)
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3.iloc[0] = np.abs(low_1d[0] - close_1d[0])   # first bar
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    up_move.iloc[0] = 0
    down_move.iloc[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and ATR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    atr_smooth = pd.Series(atr_1d).ewm(alpha=1/14, adjust=False).mean().values
    
    # Avoid division by zero
    atr_smooth = np.where(atr_smooth == 0, 1e-10, atr_smooth)
    
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate 6h Elder Ray Index
    # Bull Power = High - EMA22(close)
    # Bear Power = Low - EMA22(close)
    # Need 6h EMA22 - but we'll use 1d EMA22 aligned to 6h for smoother signal
    ema22_1d_aligned = align_htf_to_ltf(prices, df_1d, ema22_1d)
    
    bull_power = high - ema22_1d_aligned
    bear_power = low - ema22_1d_aligned
    
    # Smooth Elder Ray with 6-period EMA to reduce noise
    bull_power_s = pd.Series(bull_power)
    bear_power_s = pd.Series(bear_power)
    bull_power_smooth = bull_power_s.ewm(span=6, min_periods=6, adjust=False).mean().values
    bear_power_smooth = bear_power_s.ewm(span=6, min_periods=6, adjust=False).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive (momentum fading) or volume dries up
            if bear_power_smooth[i] > 0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative (momentum fading) or volume dries up
            if bull_power_smooth[i] < 0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Regime-based entries
                if adx[i] > 25:  # Trending regime - momentum continuation
                    # Long: Bull Power strong and rising
                    if bull_power_smooth[i] > 0 and bull_power_smooth[i] > bull_power_smooth[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short: Bear Power strong and falling (more negative)
                    elif bear_power_smooth[i] < 0 and bear_power_smooth[i] < bear_power_smooth[i-1]:
                        position = -1
                        signals[i] = -0.25
                elif adx[i] < 20:  # Ranging regime - mean reversion from extremes
                    # Long: Bear Power extremely oversold
                    if bear_power_smooth[i] < -np.std(bear_power_smooth[max(0, i-50):i]) * 1.5:
                        position = 1
                        signals[i] = 0.25
                    # Short: Bull Power extremely overbought
                    elif bull_power_smooth[i] > np.std(bull_power_smooth[max(0, i-50):i]) * 1.5:
                        position = -1
                        signals[i] = -0.25
    
    return signals