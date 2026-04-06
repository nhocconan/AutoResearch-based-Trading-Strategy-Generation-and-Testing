#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Williams Alligator + Elder Ray (Bull/Bear Power) on 12h for regime.
# Williams Alligator identifies trend presence/absence via SMAs (Jaw/Teeth/Lips).
# Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA, Bear Power = Low - EMA.
# Strategy: Only trade when Alligator indicates trending (JAW > TEETH > LIPS or reverse).
# In uptrend: buy when Bull Power turns positive (bullish momentum).
# In downtrend: sell when Bear Power turns negative (bearish momentum).
# Uses 6h timeframe for entries, 12h for regime/filter. Target: 12-35 trades/year.

name = "exp_13639_6w_alligator_elderay_12h_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13  # Smoothed
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_EMA_PERIOD = 13
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smma(data, period):
    """Smoothed Moving Average (used in Alligator)"""
    sma = np.zeros_like(data)
    sma[:period-1] = np.nan
    sma[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        sma[i] = (sma[i-1] * (period-1) + data[i]) / period
    return sma

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for regime filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Williams Alligator
    close_12h = df_12h['close'].values
    jaw = calculate_smma(close_12h, ALLIGATOR_PERIOD_JAW)
    teeth = calculate_smma(close_12h, ALLIGATOR_PERIOD_TEETH)
    lips = calculate_smma(close_12h, ALLIGATOR_PERIOD_LIPS)
    
    # Align Alligator components to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 12h Elder Ray (Bull/Bear Power)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    ema_12h = calculate_ema(close_12h, ELDER_EMA_PERIOD)
    bull_power = high_12h - ema_12h  # Bull Power = High - EMA
    bear_power = low_12h - ema_12h   # Bear Power = Low - EMA
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS, ELDER_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(atr[i]):
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
        
        # Williams Alligator trend detection
        # Uptrend: JAW > TEETH > LIPS
        # Downtrend: JAW < TEETH < LIPS
        # Otherwise: ranging/sleeping (no trade)
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        uptrend = jaw_val > teeth_val and teeth_val > lips_val
        downtrend = jaw_val < teeth_val and teeth_val < lips_val
        
        # Elder Ray signals
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Long: in uptrend AND Bull Power turns positive (bullish momentum)
        long_signal = False
        short_signal = False
        
        if i > 0:
            bull_power_prev = bull_power_aligned[i-1]
            bear_power_prev = bear_power_aligned[i-1]
            
            # Long signal: Bull Power crosses above zero in uptrend
            if uptrend and bull_power_prev <= 0 and bull_power_val > 0:
                long_signal = True
            
            # Short signal: Bear Power crosses below zero in downtrend
            if downtrend and bear_power_prev >= 0 and bear_power_val < 0:
                short_signal = True
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if trend changes or Bear Power turns negative
            if not uptrend or (i > 0 and bear_power_aligned[i-1] < 0 and bear_power_val >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if trend changes or Bull Power turns positive
            if not downtrend or (i > 0 and bull_power_aligned[i-1] > 0 and bull_power_val <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals