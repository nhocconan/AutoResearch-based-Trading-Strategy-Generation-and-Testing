#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray + 1w Trend Filter
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMMA
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
- Long: Alligator aligned bullish (Lips > Teeth > Jaw) + Bull Power > 0 + price > 1w EMA(50)
- Short: Alligator aligned bearish (Lips < Teeth < Jaw) + Bear Power < 0 + price < 1w EMA(50)
- Exit: opposite signal or stop loss (2*ATR)
- Position size: 0.25
- Target: 75-200 trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14215_6w_alligator_elder_1w_ema_v1"
timeframe = "6h"
leverage = 1.0

def smma(series, period):
    """Smoothed Moving Average (used in Williams Alligator)"""
    s = pd.Series(series)
    # First value is simple average
    sma = s.rolling(window=period, min_periods=period).mean()
    # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
    smma_vals = np.full(len(s), np.nan)
    if len(s) >= period:
        smma_vals[period-1] = sma.iloc[period-1]
        for i in range(period, len(s)):
            if not np.isnan(smma_vals[i-1]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + s.iloc[i]) / period
    return smma_vals

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_1w = calculate_ema(close_1w, 50)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (13,8,5 periods with shifts 8,5,3)
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to nan
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Elder Ray: Bull/Bear Power using EMA(13)
    ema13 = calculate_ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of Alligator setup, EMA13, ATR)
    start = max(13+8, 13, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(ema13[i]):
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
        
        # Alligator alignment
        alligator_bull = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_bear = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Elder Ray signals
        bull_energy = bull_power[i] > 0
        bear_energy = bear_power[i] < 0
        
        # 1w trend filter
        above_1w_ema = close[i] > ema_1w_aligned[i]
        below_1w_ema = close[i] < ema_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            if alligator_bull and bull_energy and above_1w_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif alligator_bear and bear_energy and below_1w_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or bearish Alligator alignment
            if close[i] <= stop_price or alligator_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or bullish Alligator alignment
            if close[i] >= stop_price or alligator_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals