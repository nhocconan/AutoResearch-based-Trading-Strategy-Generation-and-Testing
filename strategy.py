#!/usr/bin/env python3
"""
Experiment #12174: 1h Volume Spike with 4h Trend and 1d Momentum Filter
Hypothesis: On 1h timeframe, volume spikes combined with 4h EMA trend alignment and 1d RSI momentum 
capture institutional moves while avoiding chop. Using 4h for trend direction and 1d for momentum 
filter reduces false signals. Target: 60-150 trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12174_1h_volume_spike_4h_trend_1d_momentum_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_LOOKBACK = 20
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_FAST = 9
EMA_SLOW = 21
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Load 4h data ONCE before loop for trend
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop for momentum
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend
    ema_4h_fast = calculate_ema(df_4h['close'].values, EMA_FAST)
    ema_4h_slow = calculate_ema(df_4h['close'].values, EMA_SLOW)
    ema_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_fast)
    ema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slow)
    
    # Calculate 1d RSI for momentum
    rsi_1d = calculate_rsi(df_1d['close'].values, RSI_PERIOD)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike detector
    volume_ma = pd.Series(volume).rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().values
    volume_spike = volume > (volume_ma * VOLUME_SPIKE_MULTIPLIER)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_LOOKBACK, EMA_SLOW, RSI_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 4h EMA or 1d RSI not available
        if np.isnan(ema_4h_fast_aligned[i]) or np.isnan(ema_4h_slow_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
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
        
        # Trend filter (4h EMA cross)
        uptrend_4h = ema_4h_fast_aligned[i] > ema_4h_slow_aligned[i]
        downtrend_4h = ema_4h_fast_aligned[i] < ema_4h_slow_aligned[i]
        
        # Momentum filter (1d RSI)
        rsi_1d_val = rsi_1d_aligned[i]
        bullish_momentum = rsi_1d_val > RSI_OVERSOLD and rsi_1d_val < RSI_OVERBOUGHT
        bearish_momentum = rsi_1d_val > RSI_OVERSOLD and rsi_1d_val < RSI_OVERBOUGHT
        
        # Volume spike condition
        vol_spike = volume_spike[i] if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = vol_spike and uptrend_4h and bullish_momentum
        short_entry = vol_spike and downtrend_4h and bearish_momentum
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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