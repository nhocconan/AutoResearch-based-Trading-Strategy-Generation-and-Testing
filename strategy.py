#!/usr/bin/env python3
"""
4h_1d_kama_rsi_chop_v1
Strategy: 4h KAMA direction with RSI momentum and Choppiness index regime filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses KAMA for trend direction, RSI for momentum strength, and Choppiness index to filter ranging markets. Only takes trades when trend is aligned (KAMA slope), momentum confirms (RSI > 50 for long, < 50 for short), and market is trending (CHOP < 61.8). Designed to avoid whipsaws in chop while capturing trends. Target: 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI on 1d close (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Choppiness Index on 1d (14-period)
    atr_1d = np.zeros_like(close_1d)
    tr1 = np.abs(np.diff(high_1d := df_1d['high'].values, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d := df_1d['low'].values, prepend=low_1d[0]))
    tr3 = np.abs(high_1d - low_1d)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_1d * 14 / (max_high - min_low)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # KAMA slope for trend direction (using 3-period difference)
        kama_slope = kama_1d_aligned[i] - kama_1d_aligned[i-3] if i >= 3 else 0
        uptrend = kama_slope > 0
        downtrend = kama_slope < 0
        
        # RSI momentum
        rsi = rsi_1d_aligned[i]
        rsi_bullish = rsi > 50
        rsi_bearish = rsi < 50
        
        # Choppiness filter: only trade when market is trending (CHOP < 61.8)
        chop = chop_1d_aligned[i]
        trending_market = chop < 61.8
        
        # Long: uptrend + bullish RSI + trending market
        long_signal = uptrend and rsi_bullish and trending_market
        
        # Short: downtrend + bearish RSI + trending market
        short_signal = downtrend and rsi_bearish and trending_market
        
        # Exit when trend changes or RSI reverses
        exit_long = position == 1 and (not uptrend or not rsi_bullish)
        exit_short = position == -1 and (not downtrend or not rsi_bearish)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals