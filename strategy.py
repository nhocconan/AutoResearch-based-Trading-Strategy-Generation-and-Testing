#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator + Elder Ray with 1d volume confirmation and volatility filter
# Long when Green line > Red line (bullish alignment) + Bull Power > 0 + volume > 1.5x average + ATR < 50th percentile
# Short when Red line > Green line (bearish alignment) + Bear Power < 0 + volume > 1.5x average + ATR < 50th percentile
# Exit when alignment reverses or volume drops below average
# Uses Williams Alligator for trend identification, Elder Ray for bull/bear power, volume for conviction
# Designed to capture strong trends while avoiding choppy markets with volatility filter
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_WilliamsAlligator_ElderRay_VolumeVolFilter"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs with future shifts)
    # Jaw (13-period SMMA shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth (8-period SMMA shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips (5-period SMMA shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Calculate Elder Ray (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13.values
    bear_power = low - ema13.values
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Align 1d average volume to 4h
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Align Williams Alligator components to 4h
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth.values)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips.values)
    
    # Align Elder Ray components to 4h
    bull_power_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bear_power)
    
    # Align ATR to 4h
    atr_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Green > Red (bullish alignment) + Bull Power > 0 + volume spike + low volatility
            if (lips_aligned[i] > jaw_aligned[i] and  # Lips above Jaw (bullish)
                bull_power_aligned[i] > 0 and
                volume[i] > 1.5 * vol_ma_1d_aligned[i] and
                atr_aligned[i] < np.percentile(atr_aligned[max(0, i-50):i+1], 50)):  # Below median ATR
                signals[i] = 0.25
                position = 1
            # Enter short: Red > Green (bearish alignment) + Bear Power < 0 + volume spike + low volatility
            elif (jaw_aligned[i] > lips_aligned[i] and  # Jaw above Lips (bearish)
                  bear_power_aligned[i] < 0 and
                  volume[i] > 1.5 * vol_ma_1d_aligned[i] and
                  atr_aligned[i] < np.percentile(atr_aligned[max(0, i-50):i+1], 50)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: alignment reverses or volume drops
            if (lips_aligned[i] <= jaw_aligned[i]) or (volume[i] < vol_ma_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment reverses or volume drops
            if (jaw_aligned[i] <= lips_aligned[i]) or (volume[i] < vol_ma_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals