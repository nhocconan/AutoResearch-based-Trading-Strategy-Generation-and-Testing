#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (trending)
# Short when Bear Power < 0 AND Bull Power > 0 AND 12h ADX > 25 (trending)
# Uses 12h ADX to avoid ranging markets where Elder Ray gives false signals.
# Target: 12-30 trades/year on 6h. Works in bull via longs in strong uptrends and bear via shorts in strong downtrends.

name = "6h_ElderRay_12hADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF regime filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA13 and EMA26 for ADX
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_26_12h = pd.Series(close_12h).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Calculate 12h ATR13 for ADX
    tr1_12h = np.abs(high_12h[1:] - low_12h[:-1])
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_12h = pd.Series(tr_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h +DM and -DM
    up_move_12h = high_12h[1:] - high_12h[:-1]
    down_move_12h = low_12h[:-1] - low_12h[1:]
    plus_dm_12h = np.where((up_move_12h > down_move_12h) & (up_move_12h > 0), up_move_12h, 0)
    minus_dm_12h = np.where((down_move_12h > up_move_12h) & (down_move_12h > 0), down_move_12h, 0)
    plus_dm_12h = np.concatenate([[0], plus_dm_12h])
    minus_dm_12h = np.concatenate([[0], minus_dm_12h])
    
    # Calculate 12h smoothed +DM, -DM, and ATR
    plus_dm_smooth_12h = pd.Series(plus_dm_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    minus_dm_smooth_12h = pd.Series(minus_dm_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    atr_smooth_12h = pd.Series(tr_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h +DI and -DI
    plus_di_12h = np.where(atr_smooth_12h != 0, 100 * plus_dm_smooth_12h / atr_smooth_12h, 0)
    minus_di_12h = np.where(atr_smooth_12h != 0, 100 * minus_dm_smooth_12h / atr_smooth_12h, 0)
    
    # Calculate 12h DX and ADX
    dx_12h = np.where((plus_di_12h + minus_di_12h) != 0, 
                      100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 0)
    adx_12h = pd.Series(dx_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 12h ADX to 6h timeframe (wait for 12h bar to complete)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate Elder Ray components on 6h
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # Volume confirmation: 20-period volume EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema_20  # Above average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume filter
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                adx_12h_aligned[i] > 25 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 AND volume filter
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  adx_12h_aligned[i] > 25 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR ADX < 20 (regime change)
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bull Power <= 0 OR ADX < 20 (regime change)
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals