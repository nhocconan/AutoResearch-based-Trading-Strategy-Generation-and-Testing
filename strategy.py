#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d Regime Filter - Novel combination for 6h timeframe
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
- Regime: 1d ADX > 25 = trending, ADX < 20 = ranging (using 1d data)
- In trending regime (ADX>25): Trend follow - Long when Bull Power > 0 AND rising, Short when Bear Power < 0 AND falling
- In ranging regime (ADX<20): Mean revert - Long when Bear Power < -0.5*ATR AND turning up, Short when Bull Power > 0.5*ATR AND turning down
- Volume confirmation: Require volume > 1.5x 20-period average to avoid low-volume false signals
- Uses 1d ADX for HTF regime alignment to avoid whipsaws in wrong market conditions
- Designed for both bull and bear markets: regime filter adapts strategy to current market state
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime detection
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # Find first valid index
        valid_start = ~np.isnan(data)
        if not np.any(valid_start):
            return result
        first_idx = np.where(valid_start)[0][0]
        result[first_idx] = data[first_idx]
        for i in range(first_idx + 1, len(data)):
            if np.isnan(data[i]):
                result[i] = result[i-1]
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA13 for Elder Ray
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13_6h  # High - EMA13
    bear_power = low - ema13_6h   # Low - EMA13
    
    # 6h ATR for volatility adjustment in ranging regime
    tr_6h1 = np.abs(high[1:] - low[1:])
    tr_6h2 = np.abs(high[1:] - close[:-1])
    tr_6h3 = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # Need 34 for ADX, 20 for volume, 13 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema13_6h[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for new entries
            if trending_regime and volume_ok:
                # Trend following: Elder Ray momentum
                # Long: Bull Power > 0 AND rising (current > previous)
                # Short: Bear Power < 0 AND falling (current < previous)
                if i > 0:
                    bull_rising = bull_power[i] > bull_power[i-1]
                    bear_falling = bear_power[i] < bear_power[i-1]
                    
                    if bull_power[i] > 0 and bull_rising:
                        signals[i] = 0.25
                        position = 1
                    elif bear_power[i] < 0 and bear_falling:
                        signals[i] = -0.25
                        position = -1
                        
            elif ranging_regime and volume_ok:
                # Mean reversion: Elder Ray extremes
                # Long: Bear Power < -0.5*ATR AND turning up (current > previous)
                # Short: Bull Power > 0.5*ATR AND turning down (current < previous)
                if i > 0:
                    bear_turning_up = bear_power[i] > bear_power[i-1]
                    bull_turning_down = bull_power[i] < bull_power[i-1]
                    
                    if bear_power[i] < -0.5 * atr_6h[i] and bear_turning_up:
                        signals[i] = 0.25
                        position = 1
                    elif bull_power[i] > 0.5 * atr_6h[i] and bull_turning_down:
                        signals[i] = -0.25
                        position = -1
        else:
            # Manage existing position
            exit_signal = False
            
            if position == 1:
                # Exit long conditions
                if trending_regime:
                    # In trend: exit when Bull Power turns negative
                    if bull_power[i] <= 0:
                        exit_signal = True
                else:
                    # In range: exit when Bear Power reaches -0.25*ATR (profit target)
                    if bear_power[i] >= -0.25 * atr_6h[i]:
                        exit_signal = True
                        
            elif position == -1:
                # Exit short conditions
                if trending_regime:
                    # In trend: exit when Bear Power turns positive
                    if bear_power[i] >= 0:
                        exit_signal = True
                else:
                    # In range: exit when Bull Power reaches 0.25*ATR (profit target)
                    if bull_power[i] <= 0.25 * atr_6h[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0