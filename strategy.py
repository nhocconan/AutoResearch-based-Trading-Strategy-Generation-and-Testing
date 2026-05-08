#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_RSI_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA indicator parameters
    er_window = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate KAMA
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix: volatility should be rolling sum of absolute changes
    volatility = pd.Series(change).rolling(window=er_window, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index
    chop_window = 14
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=chop_window, min_periods=chop_window).mean().values
    hh = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    ll = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr.sum() / (hh - ll)) / np.log10(chop_window), 50)
    # Fix: chop calculation needs sum over window
    atr_sum = pd.Series(tr).rolling(window=chop_window, min_periods=chop_window).sum().values
    hh_ll = hh - ll
    chop = np.where(hh_ll != 0, 100 * np.log10(atr_sum / hh_ll) / np.log10(chop_window), 50)
    
    # Volume spike detection
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, chop < 61.8 (trending), volume spike
            long_cond = (close[i] > kama[i] and rsi[i] > 50 and chop[i] < 61.8 and vol_spike[i])
            
            # Short: price below KAMA, RSI < 50, chop < 61.8 (trending), volume spike
            short_cond = (close[i] < kama[i] and rsi[i] < 50 and chop[i] < 61.8 and vol_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend filter combined with RSI momentum and chop regime filter.
# Uses volume spike for confirmation. Works in both bull (trending) and bear (mean reversion in chop) markets.
# Target: 20-40 trades/year to avoid fee drag while maintaining edge.
# KAMA adapts to market noise, RSI filters exhaustion, chop identifies trending vs ranging regimes.