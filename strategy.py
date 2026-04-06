#!/usr/bin/env python3
"""
6h ADX + Williams Alligator + Volume Confirmation
Hypothesis: ADX > 25 indicates trending markets. Williams Alligator (JAW/TEETH/LIPS) provides entry signals:
- Long when LIPS > TEETH > JAW (bullish alignment) and ADX > 25
- Short when LIPS < TEETH < JAW (bearish alignment) and ADX > 25
Volume confirmation ensures trades occur during active periods.
Works in bull (trend following) and bear (trend following) by capturing directional moves.
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14455_6h_adx_alligator_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Williams Alligator (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator parameters
    jaw_period = 13   # Alligator's Jaw (blue line)
    teeth_period = 8  # Alligator's Teeth (red line)
    lips_period = 5   # Alligator's Lips (green line)
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate median price (typical price)
    median_price = (high_1d + low_1d) / 2
    
    # Jaw (Blue Line): 13-period SMA of median price, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_shift)
    
    # Teeth (Red Line): 8-period SMA of median price, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift)
    
    # Lips (Green Line): 5-period SMA of median price, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        
        tr_ma = pd.Series(tr).rolling(window=period, min_periods=period).mean()
        plus_dm_ma = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean()
        minus_dm_ma = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean()
        
        plus_di = 100 * plus_dm_ma / tr_ma
        minus_di = 100 * minus_dm_ma / tr_ma
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean()
        
        return adx.values, plus_di.values, minus_di.values
    
    adx, plus_di, minus_di = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align Alligator components and ADX to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.8 * vol_ma)  # Require at least 80% of average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(jaw_period, teeth_period, lips_period) + max(jaw_shift, teeth_shift, lips_shift) + 14 + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: Alligator alignment breaks OR ADX weakens OR stoploss
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or \
               adx_aligned[i] < 20 or \
               close[i] <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator alignment breaks OR ADX weakens OR stoploss
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or \
               adx_aligned[i] < 20 or \
               close[i] >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + ADX > 25 + volume
            bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
            bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
            strong_trend = adx_aligned[i] > 25
            
            long_setup = bullish_alignment and strong_trend and vol_filter[i]
            short_setup = bearish_alignment and strong_trend and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals