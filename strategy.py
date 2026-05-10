#!/usr/bin/env python3
# 4h_Russell_Momentum_Regime_Filter
# Hypothesis: Combine Russell 2000-inspired momentum (RSI + price > SMA) with 1d ADX regime filter to avoid chop.
# Works in bull/bear: momentum captures trends, ADX > 25 filters false signals in ranging markets.
# Target: 20-40 trades/year with disciplined entries.

name = "4h_Russell_Momentum_Regime_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # SMA(50) for trend filter
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Momentum condition: RSI > 50 and price above SMA50
    momentum_long = (rsi > 50) & (close > sma50)
    momentum_short = (rsi < 50) & (close < sma50)
    
    # 1d ADX regime filter (avoid chop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * np.divide(dm_plus_14, tr14, out=np.zeros_like(dm_plus_14), where=tr14!=0)
    di_minus = 100 * np.divide(dm_minus_14, tr14, out=np.zeros_like(dm_minus_14), where=tr14!=0)
    dx = 100 * np.divide(np.abs(di_plus - di_minus), (di_plus + di_minus), out=np.zeros_like(di_plus), where=(di_plus + di_minus)!=0)
    
    # ADX(14)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d momentum and ADX to 4h
    momentum_long_aligned = align_htf_to_ltf(prices, df_1d, momentum_long.astype(float))
    momentum_short_aligned = align_htf_to_ltf(prices, df_1d, momentum_short.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(momentum_long_aligned[i]) or np.isnan(momentum_short_aligned[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: momentum long and ADX > 25 (trending market)
            if momentum_long_aligned[i] > 0.5 and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: momentum short and ADX > 25 (trending market)
            elif momentum_short_aligned[i] > 0.5 and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: momentum fades or ADX drops below 20 (range)
            if (momentum_long_aligned[i] < 0.5) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: momentum fades or ADX drops below 20 (range)
            if (momentum_short_aligned[i] < 0.5) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals