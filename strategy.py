#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_MeanReversion_v1
Hypothesis: 4h KAMA trend direction with RSI mean reversion entries and volume confirmation.
KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI < 30 for longs, > 70 for shorts in trending markets.
Volume confirmation filters low-conviction breakouts. Discrete sizing 0.25 limits fee drag.
Designed for ~30 trades/year to avoid overtrading while capturing trends in bull/bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR for volume confirmation and stop reference
    tr1 = np.maximum(df_1d['high'].values[1:] - df_1d['low'].values[1:], np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1]))
    tr2 = np.maximum(np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1]), tr1)
    tr_1d = np.concatenate([[np.inf], tr2])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h KAMA (adaptive trend)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # temporary fix - will compute properly below
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility = pd.Series(volatility).diff(10).fillna(0).values  # 10-period sum of |change|
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed at period 10
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # already 4h
    
    # Calculate 4h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi])  # align length
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), RSI (14), ATR (14)
    start_idx = max(10, 14, 14) + 10  # extra buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * ATR (adaptive threshold)
        volume_confirm = volume[i] > 1.5 * atr_1d_aligned[i]
        
        # Determine trend from KAMA slope (rising/falling)
        if i >= 2:
            kama_slope = kama_aligned[i] - kama_aligned[i-2]
            if kama_slope > 0:
                trend = 'bullish'
            elif kama_slope < 0:
                trend = 'bearish'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'
        
        if position == 0:
            # Long setup: price above KAMA AND RSI < 30 (oversold) AND volume confirm AND bullish trend
            long_setup = (close[i] > kama_aligned[i]) and (rsi[i] < 30) and volume_confirm and (trend == 'bullish')
            
            # Short setup: price below KAMA AND RSI > 70 (overbought) AND volume confirm AND bearish trend
            short_setup = (close[i] < kama_aligned[i]) and (rsi[i] > 70) and volume_confirm and (trend == 'bearish')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR RSI > 70 (overbought) OR trend turns bearish
            if (close[i] < kama_aligned[i]) or (rsi[i] > 70) or (trend == 'bearish'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR RSI < 30 (oversold) OR trend turns bullish
            if (close[i] > kama_aligned[i]) or (rsi[i] < 30) or (trend == 'bullish'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_RSI_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0