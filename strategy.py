#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX + Williams Alligator combination
# Uses 1-day trend filter (EMA50) to align with higher timeframe direction.
# ADX > 25 indicates trending market, Alligator jaws/teeth/lips determine direction.
# Long when: ADX>25, price > Alligator teeth, and price > 1-day EMA50
# Short when: ADX>25, price < Alligator teeth, and price < 1-day EMA50
# Target: 75-150 total trades over 4 years (19-38/year) to avoid overtrading.
name = "exp_14171_6h_adx_alligator_1d_ema_v1"
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

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines"""
    median_price = (high + low) / 2
    
    # Jaws (blue line) - 13-period SMMA smoothed 8 periods ahead
    sma_jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaws = sma_jaw.shift(8)  # future shift for SMMA effect
    
    # Teeth (red line) - 8-period SMMA smoothed 5 periods ahead
    sma_teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = sma_teeth.shift(5)  # future shift for SMMA effect
    
    # Lips (green line) - 5-period SMMA smoothed 3 periods ahead
    sma_lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = sma_lips.shift(3)  # future shift for SMMA effect
    
    return jaws.values, teeth.values, lips.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Handle first element
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA(50) trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate indicators
    adx = calculate_adx(high, low, close, 14)
    jaws, teeth, lips = calculate_alligator(high, low, close)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of indicators)
    start = max(14, 13, 8, 5, 50) + 8  # +8 for Alligator jaws shift
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx[i]) or np.isnan(teeth[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]):
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
        
        # ADX + Alligator + 1d EMA signals
        # Long: ADX>25, price > teeth, price > 1d EMA50
        # Short: ADX>25, price < teeth, price < 1d EMA50
        long_signal = (adx[i] > 25) and (close[i] > teeth[i]) and (close[i] > ema_50_aligned[i])
        short_signal = (adx[i] > 25) and (close[i] < teeth[i]) and (close[i] < ema_50_aligned[i])
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when conditions reverse
            if close[i] <= stop_price or not long_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or when conditions reverse
            if close[i] >= stop_price or not short_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals