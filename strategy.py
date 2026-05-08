#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# 4h Williams Alligator + Elder Ray + Vortex with Volume Confirmation
# Uses Williams Alligator (jaw/teeth/lips) for trend direction
# Elder Ray (bull/bear power) for momentum confirmation
# Vortex indicator for trend strength
# Volume spike to filter false breakouts
# Works in bull/bear by requiring alignment across multiple timeframes
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_WilliamsAlligator_ElderRay_Vortex"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    median_price_1d = (high_1d + low_1d) / 2
    
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(median_price_1d, 13)
    teeth_raw = smma(median_price_1d, 8)
    lips_raw = smma(median_price_1d, 5)
    
    # Shift the lines
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align 1d indicators to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Vortex indicator on 4h data
    def vortex_indicator(high, low, close, period=14):
        vm_plus = np.abs(high - np.roll(low, 1))
        vm_minus = np.abs(low - np.roll(high, 1))
        tr = np.maximum(np.abs(high - low), 
                       np.maximum(np.abs(high - np.roll(close, 1)), 
                                 np.abs(low - np.roll(close, 1))))
        
        vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
        vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        vi_plus = vm_plus_sum / tr_sum
        vi_minus = vm_minus_sum / tr_sum
        return vi_plus, vi_minus
    
    vi_plus, vi_minus = vortex_indicator(high, low, close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned (lips > teeth > jaw) + Bull Power positive + VI+ > VI- + volume spike
            long_cond = (lips_aligned[i] > teeth_aligned[i] and 
                        teeth_aligned[i] > jaw_aligned[i] and
                        bull_power_aligned[i] > 0 and
                        vi_plus[i] > vi_minus[i] and
                        volume_spike[i])
            
            # Short: Alligator inverted (lips < teeth < jaw) + Bear Power negative + VI- > VI+ + volume spike
            short_cond = (lips_aligned[i] < teeth_aligned[i] and 
                         teeth_aligned[i] < jaw_aligned[i] and
                         bear_power_aligned[i] < 0 and
                         vi_minus[i] > vi_plus[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator reverses (lips < jaw) OR Bear Power negative
            if lips_aligned[i] < jaw_aligned[i] or bear_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator reverses (lips > jaw) OR Bull Power positive
            if lips_aligned[i] > jaw_aligned[i] or bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals