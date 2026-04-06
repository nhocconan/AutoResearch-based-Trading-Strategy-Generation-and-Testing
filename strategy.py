#!/usr/bin/env python3
"""
6h ADX + Williams Alligator combination with volume confirmation.
- ADX > 25 indicates trending market
- Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with future offset
- Long: Price > Teeth > Jaw AND ADX > 25 + volume > 1.5x average
- Short: Price < Teeth < Jaw AND ADX > 25 + volume > 1.5x average
- Exit: Opposite Alligator signal or stop loss (2*ATR)
- Position size: 0.25
- Target: 75-200 trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14191_6h_adx_alligator_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator lines"""
    median_price = (high + low) / 2
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    return jaw.values, teeth.values, lips.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX with proper min_periods"""
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    tr = np.zeros(len(high))
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    tr[0] = high[0] - low[0]
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.zeros(len(high))
    dx_sum = plus_di + minus_di
    dx = np.where(dx_sum != 0, 100 * np.abs(plus_di - minus_di) / dx_sum, 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h data
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of Alligator periods + ADX + volume + ATR)
    start = max(13, 8, 5, 14, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(adx_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator signals with ADX and volume filter
        # Long: Price > Teeth > Jaw AND ADX > 25 + volume
        # Short: Price < Teeth < Jaw AND ADX > 25 + volume
        alligator_long = (close[i] > teeth[i]) and (teeth[i] > jaw[i]) and (adx_1d_aligned[i] > 25) and vol_filter[i]
        alligator_short = (close[i] < teeth[i]) and (teeth[i] < jaw[i]) and (adx_1d_aligned[i] > 25) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if alligator_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif alligator_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or Alligator reversal
            if close[i] <= stop_price or not alligator_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or Alligator reversal
            if close[i] >= stop_price or not alligator_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals