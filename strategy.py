#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX Regime Filter
# Long when: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (trending market)
# Short when: Bull Power < 0 AND Bear Power > 0 AND ADX > 25 (trending market)
# Exit when: ADX < 20 (regime change to ranging) OR power signals reverse
# Elder Ray measures bull/bear power relative to EMA13
# ADX filter ensures we only trade in strong trends where Elder Ray works best
# Works in both bull and bear markets by capturing directional moves
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.25

name = "6h_ElderRay_ADXRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Regime filters
    trending_regime = adx > 25
    ranging_regime = adx < 20  # For exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(trending_regime[i]) or
            np.isnan(ranging_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Convert to bool for regime checks
        trending = bool(trending_regime[i])
        ranging = bool(ranging_regime[i])
        
        if position == 0:
            # Enter long: Bull power positive, Bear power negative, trending market
            if bull_power[i] > 0 and bear_power[i] < 0 and trending:
                signals[i] = 0.25
                position = 1
            # Enter short: Bull power negative, Bear power positive, trending market
            elif bull_power[i] < 0 and bear_power[i] > 0 and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Regime change to ranging OR power signals reverse
            if ranging or (bull_power[i] < 0) or (bear_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Regime change to ranging OR power signals reverse
            if ranging or (bull_power[i] > 0) or (bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals