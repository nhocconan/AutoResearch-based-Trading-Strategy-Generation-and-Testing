#!/usr/bin/env python3
"""
Experiment #9634: 1h EMA Pullback + 4h Trend + Volume Spike
Hypothesis: In trending markets (4h EMA21 > EMA50), pullbacks to 1h EMA21 with volume spikes offer high-probability entries.
Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend). Uses 4h for trend direction, 1h for entry timing.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag while capturing trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9634_1h_ema_pullback_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 21
EMA_SLOW = 50
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def ema(values, period):
    """Calculate EMA with proper initialization"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
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
    
    # Load HTF data ONCE before loop (4h for trend direction)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMAs for trend direction
    close_4h = df_4h['close'].values
    ema21_4h = ema(close_4h, EMA_FAST)
    ema50_4h = ema(close_4h, EMA_SLOW)
    
    # Align 4h EMAs to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMA21 for pullback entries
    ema21_1h = ema(close, EMA_FAST)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr_val = atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, 20) + 1
    
    for i in range(start, n):
        # Skip if 4h EMA data not available
        if np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine trend from 4h EMAs
        uptrend = ema21_4h_aligned[i] > ema50_4h_aligned[i]
        downtrend = ema21_4h_aligned[i] < ema50_4h_aligned[i]
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: pullback to 1h EMA21 with volume spike in direction of 4h trend
        pullback_long = uptrend and close[i] <= ema21_1h[i] and volume_spike
        pullback_short = downtrend and close[i] >= ema21_1h[i] and volume_spike
        
        # Generate signals
        if position == 0:
            if pullback_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_val[i])
            elif pullback_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_val[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals