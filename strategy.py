#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime
# Long when: Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend)
# Short when: Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 (strong trend)
# Exit when: Elder Power signals weaken (Bull Power <= 0 for long, Bear Power >= 0 for short) OR ADX < 20 (trend ends)
# Elder Ray measures bull/bear power via EMA(13); ADX filters for trending markets only
# Works in bull markets via long signals and bear markets via short signals
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

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
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = down_move[0] = np.nan
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Directional Indicators
    plus_di = 100 * pd.Series(up_move).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(down_move).rolling(window=14, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray on 6h: EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: strong trend (ADX > 25)
        strong_trend = adx_1d_aligned[i] > 25
        # Weak trend exit (ADX < 20)
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND strong trend
            if bull_power[i] > 0 and bear_power[i] < 0 and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND strong trend
            elif bear_power[i] < 0 and bull_power[i] > 0 and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR weak trend
            if bull_power[i] <= 0 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR weak trend
            if bear_power[i] >= 0 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals