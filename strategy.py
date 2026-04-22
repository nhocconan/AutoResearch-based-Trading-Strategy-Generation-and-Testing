#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1-day regime filter. Elder Ray (Bull/Bear Power) measures
# buying/selling pressure via EMA13 of (High - EMA13) and (Low - EMA13). 
# Long when Bull Power > 0 and Bear Power < 0 (bullish divergence), short when opposite.
# Uses 1-day ADX < 20 as range filter to avoid whipsaws in strong trends.
# Targets 12-37 trades/year with disciplined risk control via mean reversion to EMA13.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data for ADX regime filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ADX for regime detection
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate EMA13 for Elder Ray
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in ranging markets (ADX < 20)
        range_filter = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long entry: Bull Power > 0 and Bear Power < 0 in ranging market
            if bull_power[i] > 0 and bear_power[i] < 0 and range_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: Bull Power < 0 and Bear Power > 0 in ranging market
            elif bull_power[i] < 0 and bear_power[i] > 0 and range_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on mean reversion to EMA13
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below EMA13 (mean reversion)
                if close[i] < ema13[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above EMA13 (mean reversion)
                if close[i] > ema13[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Range_MeanReversion"
timeframe = "6h"
leverage = 1.0