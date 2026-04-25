#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing a reliable trend filter. Combined with RSI(14) for momentum confirmation and a choppiness regime filter to avoid whipsaws in ranging markets, this strategy captures strong directional moves in both bull and bear markets. Discrete position sizing (0.25) limits fee drag and drawdown. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.zeros_like(change)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1))**2
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
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr / (hh - ll)) / np.log10(period), 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for KAMA trend and RSI (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # 1d RSI for momentum
    rsi_1d = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d choppiness for regime filter
    chop_1d = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need KAMA, RSI, chop, volume MA
    start_idx = max(30, 14, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, low chop (trending), volume confirmed
            long_condition = (curr_close > kama_1d_aligned[i]) and (rsi_1d_aligned[i] > 50) and (chop_1d_aligned[i] < 50) and vol_confirmed[i]
            # Short: price < KAMA, RSI < 50, low chop (trending), volume confirmed
            short_condition = (curr_close < kama_1d_aligned[i]) and (rsi_1d_aligned[i] < 50) and (chop_1d_aligned[i] < 50) and vol_confirmed[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 40 OR chop > 61.8 (rangy)
            if (curr_close < kama_1d_aligned[i]) or (rsi_1d_aligned[i] < 40) or (chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 60 OR chop > 61.8 (rangy)
            if (curr_close > kama_1d_aligned[i]) or (rsi_1d_aligned[i] > 60) or (chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0