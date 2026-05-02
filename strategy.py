#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 indicates trending market (use Elder Ray signals), ADX < 20 indicates ranging (fade Elder Ray extremes)
# Volume confirmation (1.5x 20-period average) ensures participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime via ADX
# Uses 1d for HTF regime and Elder Ray calculation for stability

name = "6h_ElderRay_1dADXRegime_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for ADX regime and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    dm_plus[0] = dm_minus[0] = 0  # first bar
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6x volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Elder Ray, ADX and volume MA)
    start_idx = 50  # max(20 for volume, 34 for ADX/Elder Ray) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: follow Elder Ray direction
                # Long: Bull Power > 0 AND previous Bull Power <= 0 (momentum shift up)
                if (bull_power_aligned[i] > 0 and 
                    i > start_idx and bull_power_aligned[i-1] <= 0 and
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 AND previous Bear Power >= 0 (momentum shift down)
                elif (bear_power_aligned[i] < 0 and 
                      i > start_idx and bear_power_aligned[i-1] >= 0 and
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging or transition regime
                # In ranging market: fade Elder Ray extremes
                # Long: Bear Power < -0.5 * ATR(1d) AND previous Bear Power >= -0.5 * ATR(1d) (oversold bounce)
                # Short: Bull Power > 0.5 * ATR(1d) AND previous Bull Power <= 0.5 * ATR(1d) (overbought fade)
                # Approximate 1d ATR using price range since we don't have it aligned
                approx_atr_1d = (df_1d['high'].values - df_1d['low'].values)
                approx_atr_aligned = align_htf_to_ltf(prices, df_1d, approx_atr_1d)
                if not np.isnan(approx_atr_aligned[i]):
                    atr_val = approx_atr_aligned[i]
                    if (bear_power_aligned[i] < -0.5 * atr_val and 
                        i > start_idx and bear_power_aligned[i-1] >= -0.5 * atr_val and
                        volume_confirm[i]):
                        signals[i] = 0.25
                        position = 1
                    elif (bull_power_aligned[i] > 0.5 * atr_val and 
                          i > start_idx and bull_power_aligned[i-1] <= 0.5 * atr_val and
                          volume_confirm[i]):
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending long when Bull Power turns negative
                if bull_power_aligned[i] <= 0:
                    exit_signal = True
            else:
                # Exit ranging long when Bear Power rises above -0.2 * ATR (weakening oversold)
                approx_atr_1d = (df_1d['high'].values - df_1d['low'].values)
                approx_atr_aligned = align_htf_to_ltf(prices, df_1d, approx_atr_1d)
                if not np.isnan(approx_atr_aligned[i]):
                    atr_val = approx_atr_aligned[i]
                    if bear_power_aligned[i] > -0.2 * atr_val:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending short when Bear Power turns positive
                if bear_power_aligned[i] >= 0:
                    exit_signal = True
            else:
                # Exit ranging short when Bull Power falls below 0.2 * ATR (weakening overbought)
                approx_atr_1d = (df_1d['high'].values - df_1d['low'].values)
                approx_atr_aligned = align_htf_to_ltf(prices, df_1d, approx_atr_1d)
                if not np.isnan(approx_atr_aligned[i]):
                    atr_val = approx_atr_aligned[i]
                    if bull_power_aligned[i] < 0.2 * atr_val:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals