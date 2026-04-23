#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d ADX regime filter and volume confirmation.
- Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
- Regime: ADX(14) > 25 = trending (use Elder Ray), ADX <= 25 = ranging (fade extremes)
- In trending: Long when Bull Power > 0 AND rising; Short when Bear Power > 0 AND rising
- In ranging: Long when Bear Power < -0.5 * ATR(10) (oversold); Short when Bull Power < -0.5 * ATR(10) (overbought)
- Volume confirmation: 6h volume > 1.5x 20-period MA
- Discrete sizing: 0.25 to minimize fee churn, works in both bull/bear via regime adaptation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_1d = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_1d = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI, -DI, DX, ADX
    plus_di_1d = 100 * plus_dm_1d / atr_1d
    minus_di_1d = 100 * minus_dm_1d / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Rising Bull/Bear Power (current > previous)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    # ATR(10) for 6h (for oversold/overbought thresholds in ranging)
    atr_6h = pd.Series(
        np.maximum(
            np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])),
            np.abs(low[1:] - close[:-1])
        )
    ).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_6h = np.concatenate([[np.nan], atr_6h])
    
    # Volume confirmation: 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 10)  # volume MA20, EMA13, ATR10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_1d_aligned[i] > 25
        
        # Volume filter
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            if is_trending:
                # Trending regime: follow Elder Ray momentum
                # Long: Bull Power > 0 AND rising
                if bull_power[i] > 0 and bull_power_rising[i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 AND rising
                elif bear_power[i] > 0 and bear_power_rising[i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging regime: mean reversion at extremes
                # Long: Bear Power < -0.5 * ATR (oversold)
                if bear_power[i] < -0.5 * atr_6h[i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power < -0.5 * ATR (overbought)
                elif bull_power[i] < -0.5 * atr_6h[i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long: Bear Power > 0 (momentum shift) OR Bull Power < -0.5*ATR (overbought)
                if bear_power[i] > 0 or bull_power[i] < -0.5 * atr_6h[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Bull Power > 0 (momentum shift) OR Bear Power < -0.5*ATR (oversold)
                if bull_power[i] > 0 or bear_power[i] < -0.5 * atr_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0