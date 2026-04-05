#!/usr/bin/env python3
"""
Experiment #10114: 1h RSI Reversion + 4h Trend + 1d Volume Filter
Hypothesis: In mean-reverting markets (especially 2025 bear/range), RSI extremes on 1h
provide counter-trend entries when aligned with 4h trend (EMA50) and confirmed by
1d volume spikes. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Uses 4h/1d for signal direction, 1h only for entry timing. Target: 80-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10114_1h_rsi_reversion_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_TREND_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
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
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_TREND_PERIOD)
    
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume moving average for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 1h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI for mean reversion signals
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_TREND_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if 4h EMA or 1d volume MA not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
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
        
        # Trend filter: price above/below 4h EMA
        above_4h_ema = close[i] > ema_4h_aligned[i]
        below_4h_ema = close[i] < ema_4h_aligned[i]
        
        # Volume spike confirmation (1d volume > 2x 20-day average)
        volume_spike = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_SPIKE_MULTIPLIER)
        
        # Mean reversion signals: RSI extremes
        rsi_overbought = rsi[i] >= RSI_OVERBOUGHT
        rsi_oversold = rsi[i] <= RSI_OVERSOLD
        
        # Entry conditions: RSI extreme in direction opposite to 4h trend, with volume spike
        # In uptrend: look for oversold RSI to buy dips
        # In downtrend: look for overbought RSI to sell rallies
        long_entry = rsi_oversold and above_4h_ema and volume_spike
        short_entry = rsi_overbought and below_4h_ema and volume_spike
        
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