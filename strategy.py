#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_VolumeConfirm_v1
Hypothesis: Elder Ray (Bull/Bear Power) with 1d regime filter (ADX>25 = trend, ADX<20 = range) and volume confirmation captures institutional moves while avoiding whipsaws. Works in bull/bear via regime adaptation: trend follow when ADX>25, mean revert when ADX<20. Designed for 6h to target 50-150 total trades over 4 years with discrete sizing (0.25).
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
    
    # Load 1d data ONCE before loop for Elder Ray and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align to 6h (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1d ADX(14) for regime filter
    # TR calculation
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # +DM and -DM
    up_move = df_1d['high'].diff()
    down_move = df_1d['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h ATR(14) for volume confirmation threshold
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Average volume for confirmation (24-period SMA = 1d * 4 = 4d)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA(13), ADX(14), volume(24)
    start_idx = max(13, 14, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr_6h[i]
        
        # Skip if any data not ready
        if (np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(adx_val) or 
            np.isnan(avg_vol) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Regime filters
        trending = adx_val > 25
        ranging = adx_val < 20
        
        # Long conditions
        long_trend = bull_val > 0 and trending and volume_confirmed  # Bull power + trend
        long_range = bull_val > 0 and ranging and volume_confirmed and close_val < (close[i-1] if i>0 else close_val)  # Bull power + mean revert on pullback
        
        # Short conditions
        short_trend = bear_val < 0 and trending and volume_confirmed  # Bear power + trend
        short_range = bear_val < 0 and ranging and volume_confirmed and close_val > (close[i-1] if i>0 else close_val)  # Bear power + mean revert on bounce
        
        long_condition = long_trend or long_range
        short_condition = short_trend or short_range
        
        # Exit conditions
        long_exit = (position == 1 and (bull_val <= 0 or close_val < entry_price - 1.5 * atr_val))
        short_exit = (position == -1 and (bear_val >= 0 or close_val > entry_price + 1.5 * atr_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
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