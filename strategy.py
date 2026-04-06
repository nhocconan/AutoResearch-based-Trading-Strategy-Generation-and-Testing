#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12571_6d_elder_ray_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
SHORT_EMA = 13
LONG_EMA = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
REGIME_PERIOD = 50
REGIME_BULL_THRESHOLD = 0.5
REGIME_BEAR_THRESHOLD = -0.5

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

def calculate_ema_values(arr, period):
    """Calculate EMA on numpy array"""
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_sma(arr, period):
    """Calculate SMA on numpy array"""
    return pd.Series(arr).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily regime using EMA slope
    close_1d = df_1d['close'].values
    ema_fast_1d = calculate_ema_values(close_1d, 20)
    ema_slow_1d = calculate_ema_values(close_1d, 50)
    regime_raw = ema_fast_1d - ema_slow_1d
    regime_normalized = regime_raw / (pd.Series(close_1d).rolling(window=REGIME_PERIOD, min_periods=REGIME_PERIOD).std().values + 1e-10)
    regime_1d = calculate_sma(regime_normalized, 5)  # smooth regime signal
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components
    ema13 = calculate_ema_values(close, SHORT_EMA)
    ema30 = calculate_ema_values(close, LONG_EMA)
    bull_power = high - ema13
    bear_power = low - ema30
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume filter
    volume_ma = calculate_sma(volume, VOLUME_MA_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SHORT_EMA, LONG_EMA, VOLUME_MA_PERIOD, ATR_PERIOD, REGIME_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily regime not available
        if np.isnan(regime_1d_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Regime filter
        bull_regime = regime_1d_aligned[i] > REGIME_BULL_THRESHOLD
        bear_regime = regime_1d_aligned[i] < REGIME_BEAR_THRESHOLD
        
        # Elder Ray signals
        long_signal = bull_power[i] > 0 and bear_power[i] < 0 and volume_ok and bull_regime
        short_signal = bear_power[i] < 0 and bull_power[i] > 0 and volume_ok and bear_regime
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals