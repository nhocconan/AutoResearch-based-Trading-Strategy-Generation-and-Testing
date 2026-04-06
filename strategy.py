#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using KAMA trend direction with volume confirmation and 1d Bollinger Bands squeeze.
# Goes long when KAMA turns upward with above-average volume and price above 1d BB middle (SMA20) with BB width < 0.05 (squeeze),
# short when KAMA turns downward with volume and price below 1d BB middle with BB width < 0.05.
# Uses ATR-based stop loss to manage risk.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# KAMA adapts to market noise, Bollinger squeeze identifies low volatility breakout setups, volume confirms momentum.

name = "exp_13856_12h_kama_1d_bb_squeeze_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
KAMA_FAST = 2
KAMA_SLOW = 30
BB_PERIOD = 20
BB_STD = 2.0
BB_WIDTH_THRESHOLD = 0.05  # 5% width = squeeze condition
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_kama(close, fast, slow):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_bbands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    width = (upper - lower) / sma  # normalized width
    return upper, lower, width

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load 1d data for Bollinger Bands squeeze filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands for squeeze detection
    close_1d = df_1d['close'].values
    bb_upper, bb_lower, bb_width = calculate_bbands(close_1d, BB_PERIOD, BB_STD)
    
    # Align 1d Bollinger Bands to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    bb_middle_aligned = (bb_upper_aligned + bb_lower_aligned) / 2  # SMA20
    
    # 12h data for KAMA, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA for trend direction on 12h data
    kama = calculate_kama(close, KAMA_FAST, KAMA_SLOW)
    kama_up = kama > np.roll(kama, 1)  # KAMA rising
    kama_down = kama < np.roll(kama, 1)  # KAMA falling
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KAMA_SLOW, BB_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(bb_middle_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Bollinger squeeze condition (low volatility breakout setup)
        squeeze = bb_width_aligned[i] < BB_WIDTH_THRESHOLD
        
        # Price relative to BB middle
        above_bb_mid = close[i] > bb_middle_aligned[i]
        below_bb_mid = close[i] < bb_middle_aligned[i]
        
        # KAMA direction signals
        long_signal = kama_up[i] and volume_ok and squeeze and above_bb_mid
        short_signal = kama_down[i] and volume_ok and squeeze and below_bb_mid
        
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
            # Exit long on KAMA turning down
            if kama_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on KAMA turning up
            if kama_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals