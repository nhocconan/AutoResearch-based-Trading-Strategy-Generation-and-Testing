#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ADX trend filter and volume confirmation.
# Uses Alligator (Jaw/Teeth/Lips) to identify trend direction and strength.
# ADX > 25 confirms trending market; ADX < 20 indicates ranging.
# In trending markets: trade Alligator crossovers with volume confirmation.
# In ranging markets: fade extreme deviations from Alligator midline.
# Designed to work in both bull and bear markets by adapting to trend strength.
# Targets 20-40 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Williams Alligator and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (13,8,5) - Smoothed Moving Average (SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high_1d, 13)  # Blue line (13-period)
    teeth = smma(low_1d, 8)   # Red line (8-period)
    lips = smma(close_1d, 5)  # Green line (5-period)
    
    # ADX calculation (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    plus_di = np.where(tr_smooth == 0, 0, plus_di)
    minus_di = np.where(tr_smooth == 0, 0, minus_di)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 40-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_40 = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_40[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_40[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5 * 40-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Alligator conditions
        lips_above_teeth = lips_val > teeth_val
        teeth_above_jaw = teeth_val > jaw_val
        lips_below_teeth = lips_val < teeth_val
        teeth_below_jaw = teeth_val < jaw_val
        
        alligator_bullish = lips_above_teeth and teeth_above_jaw
        alligator_bearish = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Determine market regime based on ADX
            is_trending = adx_val > 25   # Strong trend
            is_ranging = adx_val < 20    # Weak trend/ranging
            
            if is_trending:
                # Trending regime: Alligator crossover with volume
                if alligator_bullish and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif alligator_bearish and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging regime: fade extreme deviations from midline
                alligator_mid = (jaw_val + teeth_val + lips_val) / 3
                deviation = (price - alligator_mid) / alligator_mid
                
                if deviation < -0.02 and vol_spike:  # 2% below midline
                    signals[i] = 0.25
                    position = 1
                elif deviation > 0.02 and vol_spike:  # 2% above midline
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on bearish Alligator setup or price below Jaw
                if alligator_bearish or price < jaw_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on bullish Alligator setup or price above Jaw
                if alligator_bullish or price > jaw_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_ADX_Volume"
timeframe = "4h"
leverage = 1.0