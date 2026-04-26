#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dADX_Regime_VolumeConfirm_v1
Hypothesis: Camarilla R1/S1 breakout with 1d ADX regime filter (trend: ADX>25, range: ADX<20) and volume confirmation. In trend, trade breakout direction; in range, fade extremes. Uses discrete sizing (0.25) to minimize fee churn. Targets 20-50 trades/year on 4h.
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
    
    # Load 1d data ONCE before loop for Camarilla levels, ADX regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low),
    #            R2 = close + 0.75*(high-low), R1 = close + 0.5*(high-low),
    #            S1 = close - 0.5*(high-low), S2 = close - 0.75*(high-low),
    #            S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We only need R1 and S1 for breakout/fade
    camarilla_range = df_1d['high'] - df_1d['low']
    camarilla_R1 = df_1d['close'] + 0.5 * camarilla_range
    camarilla_S1 = df_1d['close'] - 0.5 * camarilla_range
    
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
    
    # Align to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1.values)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1.values)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of ADX (14*3 for smoothing), volume MA (20)
    start_idx = max(14*3, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        r1_val = camarilla_R1_aligned[i]
        s1_val = camarilla_S1_aligned[i]
        adx_val = adx_14_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(adx_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: ADX>25 = trend, ADX<20 = range
        is_trend = adx_val > 25
        is_range = adx_val < 20
        
        # Trend regime: breakout of R1/S1
        # Long: price breaks above R1 with volume
        # Short: price breaks below S1 with volume
        if is_trend:
            long_condition = (close_val > r1_val) and vol_conf
            short_condition = (close_val < s1_val) and vol_conf
        # Range regime: fade extremes (mean reversion at S1/R1)
        # Long: price touches S1 and holds (bounce)
        # Short: price touches R1 and holds (rejection)
        else:
            long_condition = (close_val <= s1_val) and (close[i-1] > s1_val) and vol_conf
            short_condition = (close_val >= r1_val) and (close[i-1] < r1_val) and vol_conf
        
        # Exit: opposite condition or loss of regime
        long_exit = (position == 1 and ((is_trend and close_val < s1_val) or (is_range and close_val > r1_val)))
        short_exit = (position == -1 and ((is_trend and close_val > r1_val) or (is_range and close_val < s1_val)))
        
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

name = "4h_Camarilla_R1S1_Breakout_1dADX_Regime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0