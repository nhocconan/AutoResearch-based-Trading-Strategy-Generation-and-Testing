#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: Trade daily KAMA trend direction with RSI momentum filter and Choppiness Index regime filter.
KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI ensures momentum alignment.
Choppiness Index > 61.8 filters out strong trends where mean reversion fails, focusing on choppy regimes.
Only trade in choppy markets (CHOP > 61.8) where KAMA + RSI mean reversion edge exists.
Uses discrete sizing 0.25 to limit fee drag. Target: 7-25 trades/year to avoid fee drag.
Weekly trend filter from 1w EMA50 ensures we only trade with higher timeframe momentum.
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
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle volatility calculation correctly for rolling sum
    volatility_rolling = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=1).sum().values
    volatility_rolling = np.concatenate([[np.nan] * 9, volatility_rolling[9:]])  # align with change
    er = np.where(volatility_rolling > 0, change / volatility_rolling, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first value
    rsi = np.concatenate([[np.nan], rsi])
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/min close over 14 periods
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_close - min_close) != 0, 
                    100 * np.log10(atr_sum / (max_close - min_close)) / np.log10(14), 
                    50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(50, 30, 14, 14)  # weekly EMA50, KAMA, RSI, CHOP
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8)
        if chop[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # Determine weekly trend from EMA50
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)[i]
        if np.isnan(weekly_close_aligned):
            signals[i] = 0.0
            continue
            
        if weekly_close_aligned > ema_50_1w_aligned[i]:
            weekly_trend = 'bullish'  # bias toward longs
        elif weekly_close_aligned < ema_50_1w_aligned[i]:
            weekly_trend = 'bearish'  # bias toward shorts
        else:
            weekly_trend = 'neutral'
        
        if position == 0:
            # Long setup: price below KAMA (mean reversion long) AND RSI < 40 (oversold) AND weekly trend bullish or neutral
            long_setup = (close[i] < kama[i]) and (rsi[i] < 40) and (weekly_trend in ['bullish', 'neutral'])
            
            # Short setup: price above KAMA (mean reversion short) AND RSI > 60 (overbought) AND weekly trend bearish or neutral
            short_setup = (close[i] > kama[i]) and (rsi[i] > 60) and (weekly_trend in ['bearish', 'neutral'])
            
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
            # Exit: price crosses above KAMA OR RSI > 50 (momentum shift) OR weekly trend turns bearish
            if (close[i] > kama[i]) or (rsi[i] > 50) or (weekly_trend == 'bearish'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses below KAMA OR RSI < 50 (momentum shift) OR weekly trend turns bullish
            if (close[i] < kama[i]) or (rsi[i] < 50) or (weekly_trend == 'bullish'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0