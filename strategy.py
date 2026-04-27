#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime
- KAMA (Kaufman Adaptive Moving Average) for trend direction
- RSI for momentum confirmation
- Choppiness Index for regime filter (avoid choppy markets)
- Works in both bull/bear by using KAMA trend and avoiding whipsaws in chop
Target: 20-50 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[period:] = change[period-1:] / volatility[period-1:]
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[period-1] = close[period-1]
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: 0 = trending, 100 = choppy"""
    atr = np.zeros(len(close))
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = tr
    
    # True Range for first element
    tr0 = high[0] - low[0]
    atr[0] = tr0
    
    # Sum of ATR over period
    atr_sum = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum[i] = np.sum(atr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    max_high = np.zeros(len(close))
    min_low = np.zeros(len(close))
    for i in range(period-1, len(close)):
        max_high[i] = np.max(high[i-period+1:i+1])
        min_low[i] = np.min(low[i-period+1:i+1])
    
    # Choppiness formula
    chop = np.full(len(close), 50.0)
    for i in range(period-1, len(close)):
        if atr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    kama_1d = calculate_kama(df_1d['close'].values, period=10, fast=2, slow=30)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI on daily close
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([np.array([50.0]), rsi_1d])  # align length
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index on daily data
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Start after enough data for indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama = kama_aligned[i]
        rsi = rsi_aligned[i]
        chop = chop_aligned[i]
        
        # Only trade when market is not too choppy (chop < 61.8 = trending)
        if chop > 61.8:
            # In chop, go flat or reduce position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine KAMA trend direction
        bullish = price > kama
        bearish = price < kama
        
        # RSI thresholds for momentum
        rsi_overbought = 70
        rsi_oversold = 30
        
        if position == 0:
            # Long: price above KAMA (uptrend) and RSI not overbought
            if bullish and rsi < rsi_overbought:
                signals[i] = size
                position = 1
            # Short: price below KAMA (downtrend) and RSI not oversold
            elif bearish and rsi > rsi_oversold:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below KAMA or RSI overbought
            if not bullish or rsi >= rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price above KAMA or RSI oversold
            if not bearish or rsi <= rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0