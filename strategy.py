#!/usr/bin/env python3

"""
Hypothesis: 6-hour ADX(14) + Williams Alligator crossover with 1-week trend filter.
Long when ADX > 25 (trending) and Alligator lips cross above teeth (bullish).
Short when ADX > 25 and lips cross below teeth (bearish).
Use 1-week EMA(34) to filter trades: only long when price > weekly EMA, short when price < weekly EMA.
Exit when ADX falls below 20 (trend weakening) or Alligator lines re-cross.
Targets 12-37 trades/year (50-150 total over 4 years) with disciplined entries.
ADX filters whipsaws in ranging markets, Alligator captures trends, weekly filter avoids counter-trend trades.
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
    
    # Load 1h data for ADX and Alligator calculation - ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate ADX (14-period) from 1h data
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1h - np.roll(high_1h, 1)) > (np.roll(low_1h, 1) - low_1h), 
                       np.maximum(high_1h - np.roll(high_1h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1h, 1) - low_1h) > (high_1h - np.roll(high_1h, 1)), 
                        np.maximum(np.roll(low_1h, 1) - low_1h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM (14-period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    # Jaw (13-period, shifted 8 bars)
    jaw = pd.Series(close_1h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    # Teeth (8-period, shifted 5 bars)
    teeth = pd.Series(close_1h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    # Lips (5-period, shifted 3 bars)
    lips = pd.Series(close_1h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    
    # Align 1h indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx)
    lips_aligned = align_htf_to_ltf(prices, df_1h, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_1h, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_1h, jaw)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX trend strength filter
        strong_trend = adx_aligned[i] > 25
        weakening_trend = adx_aligned[i] < 20
        
        # Alligator crossover signals
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        
        if position == 0 and strong_trend:
            # Long: lips cross above teeth AND price above weekly EMA (uptrend filter)
            if lips_above_teeth and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: lips cross below teeth AND price below weekly EMA (downtrend filter)
            elif lips_below_teeth and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: trend weakening OR lips cross back below teeth
                if weakening_trend or lips_aligned[i] < teeth_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: trend weakening OR lips cross back above teeth
                if weakening_trend or lips_aligned[i] > teeth_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_1wEMA34"
timeframe = "6h"
leverage = 1.0