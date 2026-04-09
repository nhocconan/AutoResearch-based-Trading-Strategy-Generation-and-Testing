#!/usr/bin/env python3
# 1d_kama_rsi_chop_regime_v1
# Hypothesis: 1d strategy using KAMA for trend direction, RSI(14) for momentum/extremes, and Choppiness Index as regime filter.
# In trending markets (CHOP < 38.2): trade in direction of KAMA (long when price > KAMA, short when price < KAMA).
# In ranging markets (CHOP > 61.8): mean revert at RSI extremes (long when RSI < 30, short when RSI > 70).
# Uses volume confirmation (>1.5x 20-period average) to reduce false signals.
# Designed for low turnover (target: 30-100 total trades over 4 years) by requiring regime alignment and volume spike.
# Works in both bull and bear markets via regime adaptation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.zeros_like(change)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # For first element, set ER to 0
    er[0] = 0
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
    # Initialize KAMA with first close value
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.abs(high[0] - low[0])
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = np.where((highest_high - lowest_low) != 0,
                    100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(period),
                    50)  # Default to 50 when range is zero
    return chop

name = "1d_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1w HTF data for regime context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for long-term trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # LTF indicators
    # KAMA(10,2,30) for trend direction
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # RSI(14) for momentum/extremes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14) for regime detection
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(kama[i]) or
            np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if chop[i] > 61.8 and rsi[i] > 70:  # Range + overbought
                position = 0
                signals[i] = 0.0
            elif chop[i] < 38.2 and close[i] < kama[i]:  # Trend + price below KAMA
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if chop[i] > 61.8 and rsi[i] < 30:  # Range + oversold
                position = 0
                signals[i] = 0.0
            elif chop[i] < 38.2 and close[i] > kama[i]:  # Trend + price above KAMA
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Regime filter
                if chop[i] > 61.8:  # Ranging market
                    # Mean reversion at RSI extremes
                    if rsi[i] < 30:  # Oversold
                        position = 1
                        signals[i] = 0.25
                    elif rsi[i] > 70:  # Overbought
                        position = -1
                        signals[i] = -0.25
                elif chop[i] < 38.2:  # Trending market
                    # Follow KAMA direction with 1w trend filter
                    if close[i] > kama[i] and close[i] > ema_21_1w_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < kama[i] and close[i] < ema_21_1w_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals