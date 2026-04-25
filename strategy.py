#!/usr/bin/env python3
"""
6h Elder Ray Power + ADX Regime Filter + Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures
buying/selling pressure relative to trend. Combined with ADX regime filter (>25 = trending)
and volume confirmation, this captures strong directional moves while avoiding chop.
Works in bull/bear via ADX regime - only trades when trend is strong enough to persist.
Target: 12-30 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and power calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for Elder Ray calculation (trend baseline)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ADX for regime filter (trend strength)
    # ADX = 100 * smoothed(abs(+DI - -DI)) / (+DI + -DI)
    # Simplified: use 14-period for +DI/-DI calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_14_smoothed = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_14 / tr_14_smoothed
    minus_di = 100 * minus_dm_14 / tr_14_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray Power: Bull Power = High - EMA, Bear Power = EMA - Low
    # Calculate on 1d then align to 6h
    bull_power_1d = high_1d - ema_34_1d
    bear_power_1d = ema_34_1d - low_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        # Elder Ray signals
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Elder Ray alignment + strong trend + volume spike
            long_entry = (bull_power > 0) and strong_trend and vol_spike
            short_entry = (bear_power > 0) and strong_trend and vol_spike
            
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
            # Exit: bull power turns negative OR trend weakens (ADX < 20)
            if bull_power <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: bear power turns negative OR trend weakens (ADX < 20)
            if bear_power <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_ADX_Regime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0