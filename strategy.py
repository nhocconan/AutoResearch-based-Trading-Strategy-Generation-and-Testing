#!/usr/bin/env python3

"""
Hypothesis: Daily KAMA + RSI with Chop Filter
Trades in the direction of the daily Kaufman Adaptive Moving Average (KAMA) trend when RSI confirms momentum.
Uses weekly Choppiness Index to avoid trading in choppy markets (CHOP > 61.8) and only trade when trending (CHOP < 38.2).
Designed for low trade frequency (7-25 trades/year) to minimize fee drag and work in both bull and bear markets by
adapting to market conditions and filtering out noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_fast=2, er_slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    er = change / (volatility + 1e-10)
    er = np.where(np.isnan(er), 0, er)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr = np.abs(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))))
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    range_max_min = max_high - min_low
    chop = 100 * np.log10(atr_sum / (range_max_min + 1e-10)) / np.log10(period)
    return chop.fillna(50).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load daily data for KAMA and RSI - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily KAMA for trend
    close_1d = df_1d['close'].values
    kama = calculate_kama(close_1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI for momentum
    rsi = calculate_rsi(close_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Load weekly data for Chop filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly Chop for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    chop = calculate_chop(high_1w, low_1w, close_1w)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending = chop_aligned[i] < 38.2
        
        if position == 0 and trending:
            # Long: price above KAMA and RSI > 50 (bullish momentum)
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI < 50 (bearish momentum)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: chop becomes choppy or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: chop becomes choppy or price below KAMA
                if chop_aligned[i] >= 61.8 or close[i] < kama_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: chop becomes choppy or price above KAMA
                if chop_aligned[i] >= 61.8 or close[i] > kama_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0