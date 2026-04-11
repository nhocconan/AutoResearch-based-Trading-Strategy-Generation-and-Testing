#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on daily close
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.zeros_like(close)
        er[period:] = change[period:] / np.where(volatility[period:].sum(axis=0) != 0, volatility[period:].sum(axis=0), 1)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_values = kama(close, period=10, fast=2, slow=30)
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chopiness Index
    def chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(atr.sum(axis=0) / (highest_high - lowest_low)) / np.log10(period)
        return chop
    
    chop_values = chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(kama_values[i]) or np.isnan(rsi[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_values[i]
        rsi_val = rsi[i]
        chop_val = chop_values[i]
        
        # Chop filter: only trade when chop < 61.8 (trending market)
        if chop_val > 61.8:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price above KAMA and RSI > 50
        if price > kama_val and rsi_val > 50:
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        # Short: price below KAMA and RSI < 50
        elif price < kama_val and rsi_val < 50:
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
            position = 0
    
    return signals

# Hypothesis: KAMA trend filter + RSI momentum + Chop filter on daily timeframe.
# KAMA adapts to market noise, providing a dynamic trend line. RSI confirms momentum
# direction. Chop filter ensures we only trade in trending markets (chop < 61.8),
# avoiding whipsaws in ranging conditions. Works in both bull and bear markets by
# following the trend direction. Target: 20-50 trades over 4 years to minimize fee
# drag on daily timeframe. Discrete position sizing (0.25) reduces transaction costs.
# Uses only daily data as required, no multi-timeframe needed for this implementation.