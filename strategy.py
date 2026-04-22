#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter.
# Long when Bull Power > 0 + Bear Power < 0 + ADX > 25 (trending market)
# Short when Bear Power < 0 + Bull Power < 0 + ADX > 25
# Exit when ADX < 20 (range market) or power signals reverse.
# Works in trending markets (both bull and bear) by capturing momentum with trend strength filter.
# Avoids choppy markets where Elder Ray whipsaws. Target: 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA13 (Elder Ray) and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM (14-period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong trend)
            if bull > 0 and bear < 0 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 AND ADX > 25 (strong trend)
            elif bear < 0 and bull < 0 and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: ADX < 20 (weak trend/ranging) or power signals reverse
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when ADX weakens or Bull Power turns negative
                if adx_val < 20 or bull <= 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when ADX weakens or Bear Power turns positive
                if adx_val < 20 or bear >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime"
timeframe = "6h"
leverage = 1.0