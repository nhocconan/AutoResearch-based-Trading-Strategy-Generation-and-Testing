#!/usr/bin/env python3
"""
Experiment #10494: 1h Momentum with 4h/1d Trend Filter and Volume Spike
Hypothesis: In ranging/bear markets (2025-2026), momentum reversals on 1h timeframe 
with 4h trend alignment and 1d volume confirmation provide edge. Uses RSI(2) for 
short-term mean reversion, filtered by 4h EMA50 trend and 1d volume spike. 
Targets 15-37 trades/year (60-150 total over 4 years) by requiring confluence 
of multiple filters to avoid overtrading. Works in both bull (trend continuation) 
and bear (mean reversion in range) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10494_1h_momentum_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 2
RSI_OVERBOUGHT = 85
RSI_OVERSOLD = 15
EMA40_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA40_PERIOD)
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume average for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Align 1d volume MA to 1h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2) for short-term momentum
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA40_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Skip if 4h EMA or 1d volume MA not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
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
        
        # Volume spike condition: current 1h volume > 2.0 * 1d average volume (scaled)
        # Scale 1d volume to hourly approximation: divide by ~6.7 (24h/6 4h bars in day, but using 24 for conservative)
        volume_scaled = volume_ma_1d_aligned[i] / 4.0  # approximate hourly volume from daily
        volume_spike = volume[i] > (volume_scaled * VOLUME_SPIKE_MULTIPLIER)
        
        # RSI extreme conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Trend filter: price vs 4h EMA
        price_above_ema = close[i] > ema_4h_aligned[i]
        price_below_ema = close[i] < ema_4h_aligned[i]
        
        # Entry conditions
        long_entry = rsi_oversold and price_above_ema and volume_spike
        short_entry = rsi_overbought and price_below_ema and volume_spike
        
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