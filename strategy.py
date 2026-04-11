#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter with 12h volume confirmation
# - Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low
# - Enter long when Bull Power > 0 AND Bear Power < 0 AND ADX(12h) > 25 AND 12h volume > 1.5x 20-period volume SMA
# - Enter short when Bear Power > 0 AND Bull Power < 0 AND ADX(12h) > 25 AND 12h volume > 1.5x 20-period volume SMA
# - Exit when Elder Ray signals weaken (Bull Power < 0 for longs, Bear Power < 0 for shorts) OR ADX < 20 (regime change)
# - Uses 1d EMA13 for Elder Ray calculation, 12h for ADX and volume confirmation
# - Target: 15-30 trades/year to minimize fee drag while capturing strong trending moves with institutional participation

name = "6h_12h_1d_elder_ray_adx_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Elder Ray calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 12h data ONCE before loop for ADX and volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components for 1d
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power_1d = ema_13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 12h volume SMA for confirmation
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        volume_12h_current = df_12h['volume'].values
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_current)
        vol_confirm = volume_12h_aligned[i] > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Elder Ray signals
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # ADX regime filter
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        # Entry conditions
        enter_long = bull_power > 0 and bear_power < 0 and strong_trend and vol_confirm
        enter_short = bear_power > 0 and bull_power < 0 and strong_trend and vol_confirm
        
        # Exit conditions
        exit_long = bull_power < 0 or weak_trend  # Exit long when Bull Power weakens or trend weakens
        exit_short = bear_power < 0 or weak_trend  # Exit short when Bear Power weakens or trend weakens
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals