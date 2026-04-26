#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_VolumeConfirm_v1
Hypothesis: Elder Ray (Bull/Bear Power) with 1d regime filter (ADX>25 for trend, ADX<20 for range) and volume confirmation captures strong directional moves while avoiding chop. Works in bull/bear by adapting logic per regime. Targets 12-37 trades/year on 6h with discrete sizing (0.25).
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
    
    # Load 1d data ONCE before loop for regime and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray (Bull/Bear Power)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # 1d ADX(14) for regime filter
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    # Directional Movement
    up = df_1d['high'] - df_1d['high'].shift(1)
    down = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    # +DI, -DI, DX, ADX
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    dx_14 = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx_14).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA13 (13), ADX (14*3 for smoothing), volume MA (20)
    start_idx = max(13, 14*3, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        adx_val = adx_14_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(adx_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: ADX>25 = trend, ADX<20 = range
        is_trend = adx_val > 25
        is_range = adx_val < 20
        
        # Trend regime: Elder Ray confirms momentum
        # Long: Bull Power > 0 and rising (momentum building)
        # Short: Bear Power < 0 and falling (momentum building down)
        if is_trend:
            long_condition = (bull_val > 0) and (bull_val > bull_power_aligned[i-1]) and vol_conf
            short_condition = (bear_val < 0) and (bear_val < bear_power_aligned[i-1]) and vol_conf
        # Range regime: Elder Ray shows exhaustion at extremes
        # Long: Bear Power < 0 but improving (selling exhaustion)
        # Short: Bull Power > 0 but deteriorating (buying exhaustion)
        else:
            long_condition = (bear_val < 0) and (bear_val > bear_power_aligned[i-1]) and vol_conf
            short_condition = (bull_val > 0) and (bull_val < bull_power_aligned[i-1]) and vol_conf
        
        # Exit: opposite Elder Ray signal or loss of momentum
        long_exit = (position == 1 and ((bull_val <= 0) or (bull_val < bull_power_aligned[i-1])))
        short_exit = (position == -1 and ((bear_val >= 0) or (bear_val > bear_power_aligned[i-1])))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0