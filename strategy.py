#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI filter and 1w volatility regime filter
# KAMA adapts to market conditions - follows trends in bull markets and avoids whipsaws in sideways markets
# RSI filter prevents entries at extremes, volatility regime filter uses 1w ATR to detect high/low volatility periods
# Target: 50-150 trades over 4 years (12-38/year) with focus on quality signals
name = "exp_14124_1d_kama_rsi_vol_regime_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average with proper handling"""
    change = np.abs(close - np.roll(close, er_len))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(close) > 1 else 0
    # Handle first element
    if len(close) > er_len:
        volatility = pd.Series(close).rolling(window=er_len).apply(lambda x: np.sum(np.abs(np.diff(x, prepend=x[0]))), raw=True).values
    else:
        volatility = np.zeros_like(close)
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for volatility regime filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR for volatility regime
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    # Calculate 50-period SMA of ATR for regime classification
    atr_ma_1w = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    # Align to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA for trend
    kama = calculate_kama(close, er_len=10, fast=2, slow=30)
    
    # Calculate RSI for overbought/oversold
    rsi = calculate_rsi(close, period=14)
    
    # Calculate ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 30 for KAMA slow, 14 for RSI, 50 for ATR MA, 14 for ATR)
    start = max(30, 14, 50, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or \
           np.isnan(atr_1w_aligned[i]) or np.isnan(atr_ma_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine volatility regime: high volatility when current ATR > MA * 1.5
        high_vol_regime = atr_1w_aligned[i] > (atr_ma_1w_aligned[i] * 1.5)
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Generate signals based on KAMA trend and RSI filter
        # In low volatility regimes, we follow the trend more aggressively
        # In high volatility regimes, we require stronger signals
        
        if position == 0:
            # Long conditions: price above KAMA AND RSI not overbought
            # In high volatility, require stronger trend alignment
            if high_vol_regime:
                # In high volatility: need stronger trend confirmation
                long_condition = (close[i] > kama[i] * 1.01) and (rsi[i] < 70)
            else:
                # In low volatility: normal trend following
                long_condition = (close[i] > kama[i]) and (rsi[i] < 70)
            
            # Short conditions: price below KAMA AND RSI not oversold
            if high_vol_regime:
                # In high volatility: need stronger trend confirmation
                short_condition = (close[i] < kama[i] * 0.99) and (rsi[i] > 30)
            else:
                # In low volatility: normal trend following
                short_condition = (close[i] < kama[i]) and (rsi[i] > 30)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions for long
            # Exit if price crosses below KAMA OR RSI becomes overbought
            exit_condition = (close[i] < kama[i]) or (rsi[i] >= 70)
            if exit_condition or close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            # Exit if price crosses above KAMA OR RSI becomes oversold
            exit_condition = (close[i] > kama[i]) or (rsi[i] <= 30)
            if exit_condition or close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals