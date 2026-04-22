#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13,8,5) + 1d ADX(14) filter + volume confirmation.
# Alligator identifies trend direction and strength: jaws (13), teeth (8), lips (5).
# In trending markets (ADX > 25): trade in direction of Alligator alignment.
# In ranging markets (ADX < 20): fade extreme deviations from teeth (8-period SMMA).
# Volume spike (>1.5x 20-period average) confirms momentum.
# Designed to work in both bull and bear markets by adapting to trend strength.
# Targets 15-30 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM for ADX
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if i == 0:
                result[i] = data[i]
            elif not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
            else:
                result[i] = result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    tr_smooth = np.where(tr_smooth == 0, 1e-10, tr_smooth)
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams Alligator on 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # SMMA (Smoothed Moving Average) - equivalent to Wilder's smoothing
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_VALUE) / N
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaws (13), Teeth (8), Lips (5)
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(jaws[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Alligator alignment
        jaws_val = jaws[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Alligator aligned (all in same order) = trending
        # Jaws > Teeth > Lips = downtrend
        # Lips > Teeth > Jaws = uptrend
        alligator_up = lips_val > teeth_val and teeth_val > jaws_val
        alligator_down = jaws_val > teeth_val and teeth_val > lips_val
        
        if position == 0:
            # Determine market regime using ADX
            if adx_val > 25:  # Trending market
                # Trade in direction of Alligator alignment
                if alligator_up and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif alligator_down and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif adx_val < 20:  # Ranging market
                # Fade extreme deviations from teeth (8-period SMMA)
                # Long when price significantly below teeth
                # Short when price significantly above teeth
                deviation = (price - teeth_val) / teeth_val
                if deviation < -0.015 and vol_spike:  # 1.5% below teeth
                    signals[i] = 0.25
                    position = 1
                elif deviation > 0.015 and vol_spike:  # 1.5% above teeth
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on Alligator reversal or retracement to teeth
                if not alligator_up or price <= teeth_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on Alligator reversal or retracement to teeth
                if not alligator_down or price >= teeth_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Alligator_ADX_Volume"
timeframe = "6h"
leverage = 1.0