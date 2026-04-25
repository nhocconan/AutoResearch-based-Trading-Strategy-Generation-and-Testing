#!/usr/bin/env python3
"""
6h Elder Ray + ADX Regime Filter
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure.
Combined with ADX regime filter (ADX > 25 = trending, < 20 = ranging) to avoid whipsaws.
In trending markets: go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising.
In ranging markets: fade extremes (long when Bear Power < -threshold and turning up, short when Bull Power < -threshold and turning down).
Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years) by requiring confluence of
Elder Ray signals, ADX regime, and volume confirmation, reducing overtrading and fee drag.
Works in both bull (trend following) and bear (mean reversion in ranges) regimes.
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
    
    # Load 1d data ONCE before loop for EMA13 and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    ema_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d ADX for regime filter
    # Calculate +DI, -DI, DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Elder Ray parameters
    ema_period = 13
    bull_power_threshold = 0.0  # Bull Power > 0
    bear_power_threshold = 0.0  # Bear Power > 0 (for short)
    fade_threshold = 0.5  # Threshold for fading in ranging markets
    
    # ADX regime thresholds
    adx_trending = 25
    adx_ranging = 20
    
    # Start index: need enough for EMA and ADX calculations
    start_idx = max(30, 20)  # EMA13 needs ~25, ADX needs ~30, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Elder Ray for current bar
        bull_power = high[i] - ema_1d_aligned[i]
        bear_power = ema_1d_aligned[i] - low[i]
        
        # Calculate previous Elder Ray for momentum
        if i > start_idx:
            prev_bull_power = high[i-1] - ema_1d_aligned[i-1]
            prev_bear_power = ema_1d_aligned[i-1] - low[i-1]
        else:
            prev_bull_power = bull_power
            prev_bear_power = bear_power
        
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        
        # Determine regime
        is_trending = adx_val > adx_trending
        is_ranging = adx_val < adx_ranging
        
        if position == 0:
            # Look for entry signals
            long_entry = False
            short_entry = False
            
            if is_trending:
                # Trending regime: follow Elder Ray momentum
                # Long: Bull Power > 0 and rising
                long_entry = (bull_power > bull_power_threshold) and (bull_power > prev_bull_power) and vol_spike
                # Short: Bear Power > 0 and rising
                short_entry = (bear_power > bear_power_threshold) and (bear_power > prev_bear_power) and vol_spike
            elif is_ranging:
                # Ranging regime: fade extremes
                # Long: Bear Power < -threshold and turning up (from negative to less negative)
                long_entry = (bear_power < -fade_threshold) and (bear_power > prev_bear_power) and vol_spike
                # Short: Bull Power < -threshold and turning down (from positive to less positive)
                short_entry = (bull_power < -fade_threshold) and (bull_power < prev_bull_power) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Elder Ray deteriorates or regime changes against position
            if (bull_power <= 0) or (not is_trending and bull_power < prev_bull_power):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Elder Ray deteriorates or regime changes against position
            if (bear_power <= 0) or (not is_trending and bear_power < prev_bear_power):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0