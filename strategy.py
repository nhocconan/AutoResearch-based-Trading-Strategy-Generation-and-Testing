#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, enter long when KAMA(14,2,30) is rising and RSI(14) > 50, short when KAMA falling and RSI < 50, only when Choppiness Index(14) > 61.8 (ranging market). Uses KAMA for adaptive trend, RSI for momentum confirmation, and Choppiness to avoid whipsaws in strong trends. Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag. Works in both bull and bear markets by adapting to volatility and ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for regime filter (optional, but can add if needed)
    # For now, using 1d only as primary timeframe per instructions
    
    # KAMA calculation (adaptive moving average)
    def calculate_kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_length] = close[er_length]
        for i in range(er_length + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index calculation
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        atr_sum = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
        hh = pd.Series(high).rolling(window=length, min_periods=length).max().values
        ll = pd.Series(low).rolling(window=length, min_periods=length).min().values
        range_hl = hh - ll
        chop = 100 * np.log10(atr_sum / (range_hl * length)) / np.log10(length)
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    rsi = calculate_rsi(close, length=14)
    chop = calculate_chop(high, low, close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of KAMA(er_length), RSI, CHOP
    start_idx = max(10, 14, 14)  # er_length for KAMA, RSI length, CHOP length
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI condition: >50 for bullish momentum, <50 for bearish
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Choppiness filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop[i] > 61.8
        
        # Entry conditions
        long_entry = kama_rising and rsi_bullish and ranging_market
        short_entry = kama_falling and rsi_bearish and ranging_market
        
        # Exit conditions: opposite KAMA direction or chop < 38.2 (trending market)
        exit_long = not kama_rising or chop[i] < 38.2
        exit_short = not kama_falling or chop[i] < 38.2
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - check exit conditions
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - check exit conditions
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0