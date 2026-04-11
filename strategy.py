#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter Strategy
- KAMA determines trend direction (fast/slow adaptation)
- RSI(14) for momentum confirmation (30/70 levels)
- Chop index filter to avoid whipsaws in ranging markets (CHOP > 61.8 = range)
- Position sizing: 0.25 for long/short, 0.0 for flat
- Designed for low trade frequency (target: 20-60 trades/year) to minimize fee drag
- Works in both bull (trend following) and bear (mean reversion in range) markets
"""
from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, slow_period=10, fast_period=2):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=slow_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1))**2
    kama = np.full_like(close, np.nan)
    kama[slow_period] = close[slow_period]
    for i in range(slow_period+1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, period=14):
    """Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = tr
    atr_sum = np.zeros_like(close)
    for i in range(period, len(close)):
        atr_sum[i] = np.sum(atr[i-period+1:i+1])
    highest_high = np.zeros_like(close)
    lowest_low = np.zeros_like(close)
    for i in range(period-1, len(close)):
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    chop = np.full_like(close, 50.0)
    mask = (highest_high - lowest_low) != 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / period / (highest_high[mask] - lowest_low[mask])) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate indicators
    kama = calculate_kama(close, slow_period=10, fast_period=2)
    rsi = np.full_like(close, 50.0)
    # Calculate RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14 if i >= 1 else 0
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14 if i >= 1 else 0
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 50
    
    chop = calculate_chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if KAMA not ready
        if np.isnan(kama[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Market regime filter: avoid trading in strong ranging markets
        # Chop > 61.8 indicates ranging market (avoid trend following)
        # Chop < 38.2 indicates trending market (favor trend following)
        chop_value = chop[i] if not np.isnan(chop[i]) else 50
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # KAMA trend direction
        kama_up = kama[i] > kama[i-1] if i > 0 else False
        kama_down = kama[i] < kama[i-1] if i > 0 else False
        
        # RSI momentum
        rsi_value = rsi[i] if not np.isnan(rsi[i]) else 50
        rsi_overbought = rsi_value > 70
        rsi_oversold = rsi_value < 30
        
        # Entry logic: adapt to market regime
        long_entry = False
        short_entry = False
        
        if is_trending:
            # In trending markets: follow KAMA direction with RSI filter
            long_entry = kama_up and rsi_value < 70  # Not overbought
            short_entry = kama_down and rsi_value > 30  # Not oversold
        elif is_ranging:
            # In ranging markets: mean reversion at RSI extremes
            long_entry = rsi_oversold and kama_up  # Oversold + turning up
            short_entry = rsi_overbought and kama_down  # Overbought + turning down
        else:
            # Transition zone: neutral, no entries
            pass
        
        # Exit logic: opposite signal or RSI extreme in trending markets
        long_exit = False
        short_exit = False
        
        if is_trending:
            # Exit on opposite KAMA direction
            long_exit = kama_down
            short_exit = kama_up
        else:
            # Exit on RSI normalization
            long_exit = rsi_value >= 50
            short_exit = rsi_value <= 50
        
        # Update position
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals