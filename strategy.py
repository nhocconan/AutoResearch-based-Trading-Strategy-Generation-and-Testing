#!/usr/bin/env python3
"""
6h_ADX_Trend_ElderRay_Pullback_v1
Hypothesis: On 6h timeframe, use 1d ADX(14) > 25 for trending regime, and Elder Ray (Bull/Bear Power) pullback to EMA13 for entry.
Long when ADX > 25, Bull Power > 0, and close pulls back to EMA13 (close <= EMA13 * 1.005).
Short when ADX > 25, Bear Power < 0, and close pulls back to EMA13 (close >= EMA13 * 0.995).
Exit when trend weakens (ADX < 20) or Elder Power reverses.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.
Works in both bull and bear markets by only trading in strong trends with pullback entries.
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA13 on 1d close for pullback reference
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align HTF indicators to 6h timeframe (completed 1d bar lag)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=1)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d, additional_delay_bars=1)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power, additional_delay_bars=1)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX and EMA
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema13_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for pullback entries in direction of 1d trend
            # Long: ADX > 25 (strong trend), Bull Power > 0 (bulls in control), pullback to EMA13
            # Short: ADX > 25 (strong trend), Bear Power < 0 (bears in control), pullback to EMA13
            long_signal = (adx_aligned[i] > 25) and (bull_power_aligned[i] > 0) and (close[i] <= ema13_aligned[i] * 1.005)
            short_signal = (adx_aligned[i] > 25) and (bear_power_aligned[i] < 0) and (close[i] >= ema13_aligned[i] * 0.995)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when trend weakens or bulls lose control
            exit_signal = (adx_aligned[i] < 20) or (bull_power_aligned[i] <= 0)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when trend weakens or bears lose control
            exit_signal = (adx_aligned[i] < 20) or (bear_power_aligned[i] >= 0)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Trend_ElderRay_Pullback_v1"
timeframe = "6h"
leverage = 1.0