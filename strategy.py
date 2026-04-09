#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Elder Ray (Bull/Bear Power) measures bull/bear strength relative to EMA13
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Go long when Bull Power > 0 and Bear Power rising (momentum)
# Go short when Bear Power < 0 and Bull Power falling (momentum)
# Use 1d ADX > 25 to filter for trending markets only (avoid chop)
# Position size 0.25 to limit drawdown
# Target: 20-40 trades/year per symbol to minimize fee drag

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm_1d = np.zeros(len(df_1d))
    minus_dm_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm_1d[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_1d[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed TR, +DM, -DM (14-period Wilder smoothing)
    tr_sum_1d = np.zeros(len(df_1d))
    plus_dm_sum_1d = np.zeros(len(df_1d))
    minus_dm_sum_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i < 14:
            if i == 0:
                tr_sum_1d[i] = tr_1d[i]
                plus_dm_sum_1d[i] = plus_dm_1d[i]
                minus_dm_sum_1d[i] = minus_dm_1d[i]
            else:
                tr_sum_1d[i] = tr_sum_1d[i-1] + tr_1d[i]
                plus_dm_sum_1d[i] = plus_dm_sum_1d[i-1] + plus_dm_1d[i]
                minus_dm_sum_1d[i] = minus_dm_sum_1d[i-1] + minus_dm_1d[i]
        else:
            tr_sum_1d[i] = tr_sum_1d[i-1] - (tr_sum_1d[i-1] / 14) + tr_1d[i]
            plus_dm_sum_1d[i] = plus_dm_sum_1d[i-1] - (plus_dm_sum_1d[i-1] / 14) + plus_dm_1d[i]
            minus_dm_sum_1d[i] = minus_dm_sum_1d[i-1] - (minus_dm_sum_1d[i-1] / 14) + minus_dm_1d[i]
    
    # Directional Indicators
    plus_di_1d = np.zeros(len(df_1d))
    minus_di_1d = np.zeros(len(df_1d))
    dx_1d = np.zeros(len(df_1d))
    
    for i in range(14, len(df_1d)):
        if tr_sum_1d[i] != 0:
            plus_di_1d[i] = (plus_dm_sum_1d[i] / tr_sum_1d[i]) * 100
            minus_di_1d[i] = (minus_dm_sum_1d[i] / tr_sum_1d[i]) * 100
            if (plus_di_1d[i] + minus_di_1d[i]) != 0:
                dx_1d[i] = (abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i])) * 100
    
    # ADX (smoothed DX)
    adx_1d = np.zeros(len(df_1d))
    dx_sum = 0.0
    for i in range(len(df_1d)):
        if i < 27:  # First 14 DX + 13 smoothing = 27
            if i >= 14:
                dx_sum += dx_1d[i]
        else:
            if i == 27:
                dx_sum = np.sum(dx_1d[14:28])  # First 14 DX values
            else:
                dx_sum = dx_sum - (dx_sum / 14) + dx_1d[i]
            if i >= 27:
                adx_1d[i] = dx_sum
    
    # Align 1d ADX to 6h timeframe (only use completed daily bars)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate EMA13 on 6h for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Momentum of Elder Ray (1-period change)
    bull_power_mom = bull_power - np.roll(bull_power, 1)
    bear_power_mom = bear_power - np.roll(bear_power, 1)
    bull_power_mom[0] = 0
    bear_power_mom[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after EMA13 warmup
        # Skip if ADX not available
        if np.isnan(adx_6h[i]):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_6h[i] <= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive (bulls losing control)
            if bear_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative (bears losing control)
            if bull_power[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Bull Power > 0 and rising, Bear Power < 0
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and
                bull_power_mom[i] > 0):
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power < 0 and falling, Bull Power > 0
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and
                  bear_power_mom[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals