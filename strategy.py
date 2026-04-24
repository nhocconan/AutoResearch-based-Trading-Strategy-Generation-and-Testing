#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 12h ADX regime filter.
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
- Regime filter: Only trade when 12h ADX > 25 (trending market)
- Long when Bull Power > 0 AND rising ( Bull Power > Bull Power 1 bar ago )
- Short when Bear Power > 0 AND rising ( Bear Power > Bear Power 1 bar ago )
- Uses discrete size 0.25 to limit drawdown in 2022 crash
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
- Works in bull/bear: ADX regime ensures we only trade strong trends, Elder Ray captures momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMA13 for Elder Ray (using 6h close)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Higher = stronger bulls
    bear_power = ema13 - low   # Higher = stronger bears
    
    # Calculate 12h ADX for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_12h[0] - low_12h[0]  # First bar
    atr12 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_12h - np.roll(high_12h, 1))
    down_move = pd.Series(np.roll(low_12h, 1) - low_12h)
    up_move.iloc[0] = 0
    down_move.iloc[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(atr12).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Regime filter: only trade when ADX > 25 (trending)
    trending = adx_aligned > 25
    
    # Elder Ray signals: rising power indicates strengthening momentum
    bull_rising = bull_power > np.roll(bull_power, 1)
    bear_rising = bear_power > np.roll(bear_power, 1)
    bull_rising[0] = False
    bear_rising[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30)  # Need EMA13 and enough for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(trending[i]) or np.isnan(bull_rising[i]) or np.isnan(bear_rising[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND rising AND trending regime
            if bull_power[i] > 0 and bull_rising[i] and trending[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND rising AND trending regime
            elif bear_power[i] > 0 and bear_rising[i] and trending[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power falls below 0 OR loses momentum OR regime changes
            if bull_power[i] <= 0 or not bull_rising[i] or not trending[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power falls below 0 OR loses momentum OR regime changes
            if bear_power[i] <= 0 or not bear_rising[i] or not trending[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hADX_Regime_v1"
timeframe = "6h"
leverage = 1.0