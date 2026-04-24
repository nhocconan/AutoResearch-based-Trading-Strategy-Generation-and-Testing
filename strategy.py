#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for regime filter (trending when ADX > 25, ranging when ADX < 20).
- Entry: Elder Ray Bull Power > 0 and Bear Power < 0 with volume > 1.5 * 20-period volume MA and aligned with 1d regime.
- Exit: Opposite Elder Ray signal or ATR-based stop (2.0 * ATR(6h)).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by using 1d ADX regime to avoid whipsaws in ranging markets and Elder Ray for trend strength confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Elder Ray and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h = df_6h['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Bull Power and Bear Power
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    bull_power = high_6h - ema_13
    bear_power = low_6h - ema_13
    
    # Calculate 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_6h, ema_13)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(13, 14, 20, 14, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(atr_6h_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # 1d ADX regime: trending when ADX > 25, ranging when ADX < 20
        adx_trending = adx_aligned[i] > 25
        adx_ranging = adx_aligned[i] < 20
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
            
            # Long: Bull Power > 0, Bear Power < 0, volume confirmed, and 1d trending regime
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                vol_confirmed and adx_trending):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bull Power < 0, Bear Power > 0, volume confirmed, and 1d trending regime
            elif (bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0 and 
                  vol_confirmed and adx_trending):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or Elder Ray reversal (Bear Power > 0)
            stop_loss = entry_price - 2.0 * atr_6h_aligned[i]
            if curr_low < stop_loss or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or Elder Ray reversal (Bull Power < 0)
            stop_loss = entry_price + 2.0 * atr_6h_aligned[i]
            if curr_high > stop_loss or bull_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0