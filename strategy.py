#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: Daily KAMA trend direction + RSI mean reversion + choppy market filter captures swings in both bull and bear regimes. Uses 1d primary timeframe with 1w trend filter for alignment. Targets 30-100 trades over 4 years by requiring confluence of trend, momentum, and regime filters. Discrete sizing (0.25) minimizes fee drag.
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter (more responsive than EMA50)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14) for mean reversion
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Chopiness Index(14) for regime filter
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Chop = 100 * log10(sumTR / (ATR * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA seed(10), RSI(14), ATR(14), Chop(14)
    start_idx = max(10, 14)
    
    for i in range(start_idx, n):
        kama_val = kama[i]
        close_val = close[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_val = ema_34_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(close_val) or np.isnan(rsi_val) or 
            np.isnan(chop_val) or np.isnan(ema_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs weekly EMA34
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # RSI mean reversion: oversold/overbought
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        # Chop filter: only trade in ranging markets (chop > 50)
        chop_ranging = chop_val > 50
        
        # Long: price > KAMA (trend up) + RSI oversold + choppy market
        long_condition = (close_val > kama_val) and rsi_oversold and chop_ranging
        # Short: price < KAMA (trend down) + RSI overbought + choppy market
        short_condition = (close_val < kama_val) and rsi_overbought and chop_ranging
        
        # Exit: RSI returns to neutral zone (40-60)
        long_exit = (position == 1 and rsi_val > 40)
        short_exit = (position == -1 and rsi_val < 60)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0