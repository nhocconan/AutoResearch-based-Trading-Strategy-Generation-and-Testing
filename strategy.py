#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12731_6h_obv_ma_crossover_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
OBV_MA_FAST = 10
OBV_MA_SLOW = 30
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    return (np.sign(np.diff(close, prepend=close[0])) * volume).cumsum()

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 6h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate OBV and its moving averages
    obv = calculate_obv(close, volume)
    obv_ma_fast = pd.Series(obv).ewm(span=OBV_MA_FAST, adjust=False, min_periods=OBV_MA_FAST).mean().values
    obv_ma_slow = pd.Series(obv).ewm(span=OBV_MA_SLOW, adjust=False, min_periods=OBV_MA_SLOW).mean().values
    
    # Calculate ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(OBV_MA_FAST, OBV_MA_SLOW, ATR_PERIOD) + 1
    
    for i in range(start, n):
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
        
        # Volume confirmation: current volume above average
        volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()[i]
        volume_ok = not np.isnan(volume_ma) and volume[i] > (volume_ma * VOLUME_THRESHOLD)
        
        # OBV crossover signals
        obv_cross_up = obv_ma_fast[i] > obv_ma_slow[i] and obv_ma_fast[i-1] <= obv_ma_slow[i-1]
        obv_cross_down = obv_ma_fast[i] < obv_ma_slow[i] and obv_ma_fast[i-1] >= obv_ma_slow[i-1]
        
        # Generate signals
        if position == 0:
            if obv_cross_up and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif obv_cross_down and volume_ok:
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