#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator combined with Elder Ray Power for trend strength
# Uses 6h as primary timeframe with 12h trend filter. Alligator (SMAs) identifies trend direction
# and alignment, while Elder Ray (bull/bear power) measures trend strength behind price moves.
# Works in bull/bear by capturing strong trending moves while avoiding weak/choppy markets.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and cost.

name = "elder_alligator_6h_12h_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW = 13   # Smoothed SMMA(13)
ALLIGATOR_TEETH = 8  # Smoothed SMMA(8)
ALLIGATOR_LIPS = 5   # Smoothed SMMA(5)
ELDER_POWER_PERIOD = 13
EMA_TREND_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def smma(close, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_elder_power(high, low, close, period):
    """Calculate Elder Ray Power: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_TREND_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (3 SMMA lines)
    jaw = smma(close, ALLIGATOR_JAW)
    teeth = smma(close, ALLIGATOR_TEETH)
    lips = smma(close, ALLIGATOR_LIPS)
    
    # Elder Ray Power
    bull_power, bear_power = calculate_elder_power(high, low, close, ELDER_POWER_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW, ALLIGATOR_TEETH, ALLIGATOR_LIPS, ELDER_POWER_PERIOD, EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator alignment: check if lines are properly ordered
        # In uptrend: Lips > Teeth > Jaw
        # In downtrend: Lips < Teeth < Jaw
        alligator_bull = (lips[i] > teeth[i] > jaw[i]) if not (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])) else False
        alligator_bear = (lips[i] < teeth[i] < jaw[i]) if not (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])) else False
        
        # Elder Ray confirmation: strong bull/bear power
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1]) if not np.isnan(bull_power[i]) else False
        strong_bear = bear_power[i] < 0 and abs(bear_power[i]) > np.abs(np.mean(bear_power[max(0, i-20):i+1])) if not np.isnan(bear_power[i]) else False
        
        # Trend filter from 12h EMA
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Entry signals require: Alligator alignment + Elder Ray power + 12h trend agreement
        bullish_setup = alligator_bull and strong_bull and uptrend_12h
        bearish_setup = alligator_bear and strong_bear and downtrend_12h
        
        # Generate signals
        if position == 0:
            if bullish_setup:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_setup:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals