#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power + 12h ADX regime filter for trend strength.
- Primary timeframe: 6h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 12h ADX(14) to filter trending (ADX > 25) vs ranging (ADX < 20) markets.
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low.
- Entry: Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (strong uptrend).
         Short when Bull Power < 0 AND Bear Power > 0 AND 12h ADX > 25 (strong downtrend).
- Exit: Reverse signal or when 12h ADX < 20 (trend weakening).
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures trend momentum with institutional power metrics and avoids 
whipsaws in ranging markets via ADX regime filter, working in both bull and bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for ADX regime filter and EMA13
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA13 for Elder Ray Power
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h ADX(14) for trend strength
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = df_12h['high'] - df_12h['high'].shift(1)
    down_move = df_12h['low'].shift(1) - df_12h['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = df_12h['high'].values - ema_12h
    bear_power = ema_12h - df_12h['low'].values
    
    # Align HTF indicators to 6h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14)  # Need enough bars for ADX and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals with ADX > 25 (strong trend)
            strong_trend = adx_aligned[i] > 25
            
            # Long: Bull Power > 0 AND Bear Power < 0 AND strong uptrend
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND strong downtrend
            elif bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0 and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when trend weakens (ADX < 20) or reverse signal
            if adx_aligned[i] < 20 or (bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when trend weakens (ADX < 20) or reverse signal
            if adx_aligned[i] < 20 or (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_12hADX_Regime_v1"
timeframe = "6h"
leverage = 1.0