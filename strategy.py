#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: Trade KAMA directional signals on 12h with RSI momentum filter and Choppiness Index regime filter. 
Long when KAMA turns up + RSI>50 + CHOP<61.8 (trending market); Short when KAMA turns down + RSI<50 + CHOP<61.8.
Avoids ranging markets (CHOP>61.8) where trend signals fail. Uses discrete position sizing (0.25) to minimize fee drag.
Designed for 12h timeframe targeting 12-37 trades/year with strong edge in both bull and bear regimes.
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend direction
    # ER = |net change| / sum(|price changes|) over lookback
    # Smooth = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prev_KAMA + smooth * (price - prev_KAMA)
    lookback = 10
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    net_change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if lookback == 1 else \
                 pd.Series(close).rolling(window=lookback).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    # Handle first lookback bars
    volatility = np.concatenate([np.full(lookback-1, np.nan), volatility[lookback-1:]])
    er = np.where(volatility > 0, net_change / volatility, 0)
    er = np.concatenate([np.full(lookback-1, np.nan), er])
    
    # Calculate smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[lookback-1] = close[lookback-1]  # seed
    for i in range(lookback, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) for momentum filter
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])  # align with close
    
    # Calculate Choppiness Index(14) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(lookback)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with high/low
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA(10), RSI(14), CHOP(14)
    start_idx = max(10, 14, 14) + 5  # extra buffer
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine KAMA direction (trend)
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Momentum filter
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Regime filter: avoid ranging markets
        trending_market = chop[i] < 61.8
        
        if position == 0:
            # Look for KAMA turning points with momentum and regime confirmation
            long_signal = kama_rising and rsi_bullish and trending_market
            short_signal = kama_falling and rsi_bearish and trending_market
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when KAMA turns down OR momentum fades OR market becomes ranging
            exit_signal = (not kama_rising) or (rsi[i] < 40) or (chop[i] > 61.8)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when KAMA turns up OR momentum fades OR market becomes ranging
            exit_signal = (not kama_falling) or (rsi[i] > 60) or (chop[i] > 61.8)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0