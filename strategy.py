#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike + Regime Filter (ADX)
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction, Elder Ray (Bull/Bear Power) for momentum,
# volume spike for confirmation, and ADX for regime filtering (ADX > 25 = trending).
# Designed to work in both bull and bear markets by only taking trades in strong trends.
# Targets 15-30 trades/year on 12h timeframe with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Williams Alligator and Elder Ray (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (13,8,5) - Smoothed with SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            sma = np.nansum(arr[:period]) / period
            result[period-1] = sma
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    # Elder Ray - Bull Power and Bear Power
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # ADX for regime filtering (14-period)
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
    plus_di = np.where(np.isnan(plus_di), 0, plus_di)
    minus_di = np.where(np.isnan(minus_di), 0, minus_di)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Williams Alligator condition: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        lips = lips_aligned[i]
        teeth = teeth_aligned[i]
        jaw = jaw_aligned[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        
        # ADX filter: only trade in trending markets (ADX > 25)
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw AND Bull Power > 0 AND Volume Spike
            if lips > teeth and teeth > jaw and bull_power > 0 and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw AND Bear Power < 0 AND Volume Spike
            elif lips < teeth and teeth < jaw and bear_power < 0 and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Lips cross below Teeth OR Bear Power becomes positive
                if lips < teeth or bear_power > 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Lips cross above Teeth OR Bull Power becomes negative
                if lips > teeth or bull_power < 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Alligator_ElderRay_Volume_Trend"
timeframe = "12h"
leverage = 1.0