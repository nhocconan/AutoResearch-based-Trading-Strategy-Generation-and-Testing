#!/usr/bin/env python3
"""
Experiment #9594: 1h Momentum Reversal with 4h/1d Trend Filter.
Hypothesis: In strong trends (4h EMA21 aligned with 1d EMA50), 1h RSI extremes (>70/<30) 
with volume confirmation provide high-probability continuation entries. 
In ranging markets (4h EMA21 crossing 1d EMA50), RSI extremes signal reversals.
Uses 4h/1d for trend regime and direction, 1h only for entry timing.
Target: 60-150 trades over 4 years (15-37/year) to avoid fee drag.
Works in bull (continuation in uptrend) and bear (continuation in downtrend) with 
mean reversion in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9594_1h_momentum_reversal_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_FAST = 21  # 4h trend
EMA_SLOW = 50  # 1d trend
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    ema_4h = calculate_ema(close_4h, EMA_FAST)
    ema_1d = calculate_ema(close_1d, EMA_SLOW)
    
    # Align HTF indicators to LTF
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_SLOW, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        
        # Trend regime: 4h EMA21 vs 1d EMA50
        uptrend = ema_4h_aligned[i] > ema_1d_aligned[i]
        downtrend = ema_4h_aligned[i] < ema_1d_aligned[i]
        ranging = np.abs(ema_4h_aligned[i] - ema_1d_aligned[i]) / ema_1d_aligned[i] < 0.01  # within 1%
        
        # Volume confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # RSI conditions
        rsi_overbought = rsi[i] >= RSI_OVERBOUGHT
        rsi_oversold = rsi[i] <= RSI_OVERSOLD
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if uptrend:
            # In uptrend: buy RSI oversold pullbacks with volume
            long_entry = rsi_oversold and volume_spike
        elif downtrend:
            # In downtrend: sell RSI overbought bounces with volume
            short_entry = rsi_overbought and volume_spike
        else:  # ranging
            # In range: fade RSI extremes
            long_entry = rsi_oversold and volume_spike
            short_entry = rsi_overbought and volume_spike
        
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