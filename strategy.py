#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator + Elder Ray + Vortex with 1w trend filter
# Uses Williams Alligator (SMAs) for trend direction, Elder Ray for bull/bear power,
# and Vortex indicator for trend confirmation. Long when green line above red,
# bull power > 0, bear power < 0, and VI+ > VI-. Short when opposite conditions.
# Weekly trend filter ensures alignment with higher timeframe trend.
# Position size: 0.25 to limit drawdown. Target: 15-25 trades/year.

name = "1d_Williams_Alligator_ElderRay_Vortex_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_1w = df_1w['close']
    ema_21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_prev = np.roll(ema_21_1w, 1)
    ema_21_1w_prev[0] = ema_21_1w[0]
    ema_rising_1w = ema_21_1w > ema_21_1w_prev
    ema_falling_1w = ema_21_1w < ema_21_1w_prev
    ema_rising_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_rising_1w)
    ema_falling_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_falling_1w)
    
    # Williams Alligator: SMA(13,8), SMA(8,5), SMA(5,3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Align Alligator lines
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Vortex Indicator: VI+ and VI-
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or
            np.isnan(ema_rising_1w_aligned[i]) or np.isnan(ema_falling_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: lips > teeth (green above red), bull power > 0, bear power < 0, VI+ > VI-, weekly trend up
            if (lips_aligned[i] > teeth_aligned[i] and 
                bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                vi_plus[i] > vi_minus[i] and
                ema_rising_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: lips < teeth (green below red), bull power < 0, bear power > 0, VI- > VI+, weekly trend down
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  vi_minus[i] > vi_plus[i] and
                  ema_falling_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: lips < teeth OR weekly trend turns down
            if (lips_aligned[i] < teeth_aligned[i]) or (not ema_rising_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: lips > teeth OR weekly trend turns up
            if (lips_aligned[i] > teeth_aligned[i]) or (not ema_falling_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals