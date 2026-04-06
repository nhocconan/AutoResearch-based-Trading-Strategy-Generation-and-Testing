#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12699_6d_keltner_channel_vol_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
KELTNER_MULT = 2.0
EMA_PERIOD = 20
ATR_PERIOD = 10
VOLUME_MA_PERIOD = 20
VOLUME_BREAKOUT_MULT = 3.0
SIGNAL_SIZE = 0.25
ATR_STOP_MULTIPLIER = 2.5

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

def calculate_keltner_channels(high, low, close, ema_period, atr_period, multiplier):
    """Calculate Keltner Channel upper and lower bands"""
    ema = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    return upper, lower, ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel
    keltner_upper, keltner_lower, keltner_middle = calculate_keltner_channels(
        high, low, close, EMA_PERIOD, ATR_PERIOD, KELTNER_MULT
    )
    
    # Volume filter
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, ATR_PERIOD, VOLUME_MA_PERIOD, 200) + 1
    
    for i in range(start, n):
        # Skip if daily EMA200 not available
        if np.isnan(ema200_1d_aligned[i]):
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
        
        # Volume breakout condition
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_BREAKOUT_MULT) if not np.isnan(volume_ma[i]) else False
        
        # Determine trend from daily EMA200
        uptrend = close[i] > ema200_1d_aligned[i]
        downtrend = close[i] < ema200_1d_aligned[i]
        
        # Keltner breakout with volume and trend alignment
        breakout_long = volume_ok and close[i] > keltner_upper[i] and uptrend
        breakout_short = volume_ok and close[i] < keltner_lower[i] and downtrend
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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