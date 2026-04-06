#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray combination with 1d trend filter.
- Williams Alligator: Jaw (13-period smoothed), Teeth (8-period smoothed), Lips (5-period smoothed)
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
- Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND price > 1d EMA(50)
- Short: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND price < 1d EMA(50)
- Exit: Opposite Alligator signal or 2*ATR stop
- Position size: 0.25
- Target: 75-200 trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14235_6h_alligator_elder_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_sma(close, period):
    """Calculate SMA with proper min_periods"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_smma(close, period):
    """Calculate Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    sma = calculate_sma(close, period)
    smma = np.full_like(sma, np.nan)
    if len(sma) >= period:
        smma[period-1] = sma[period-1]
        for i in range(period, len(sma)):
            smma[i] = (smma[i-1] * (period-1) + close[i]) / period
    return smma

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (13, 8, 5) - using SMMA
    jaw = calculate_smma(close, 13)  # Jaw (13-period)
    teeth = calculate_smma(close, 8)  # Teeth (8-period)
    lips = calculate_smma(close, 5)   # Lips (5-period)
    
    # Elder Ray - using EMA(13)
    ema13 = calculate_ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 13 for Alligator, 13 for EMA13, 50 for 1d EMA)
    start = max(13, 13, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]):
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
        
        # Alligator signals
        # Bullish: Lips > Teeth > Jaw
        alligator_bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish: Lips < Teeth < Jaw
        alligator_bearish = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Elder Ray signals
        bullish_elder = bull_power[i] > 0
        bearish_elder = bear_power[i] < 0
        
        # 1d EMA filter
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            if alligator_bullish and bullish_elder and price_above_1d_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif alligator_bearish and bearish_elder and price_below_1d_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or bearish Alligator signal
            if close[i] <= stop_price or alligator_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or bullish Alligator signal
            if close[i] >= stop_price or alligator_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals