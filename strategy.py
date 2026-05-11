#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Chop_Filter
Hypothesis: Uses KAMA on 12h for trend direction, RSI(14) for momentum, and Choppiness Index for regime filtering.
Trades only in trending markets (CHOP < 38.2) in the direction of KAMA when RSI confirms momentum.
Avoids choppy markets to reduce whipsaw. Designed for low trade frequency (<20/year) to avoid fee drag.
"""

name = "12h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Choppiness Index (needs daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- KAMA on 12h for trend direction ---
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # Sum of |close[i] - close[i-1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = kama  # already aligned to 12h
    
    # --- Choppiness Index on 1d ---
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Chop = 100 * log10(sumTR / (ATR * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr / (atr * 14)) / np.log10(14)
    chop = chop  # already aligned to 1d
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # --- RSI(14) on 12h ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first 14 values as NaN to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(rsi[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending = chop_aligned[i] < 38.2
        
        # Trend direction: price vs KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum confirmation
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        rsi_bullish = 50 < rsi[i] < 70  # rising momentum
        rsi_bearish = 30 < rsi[i] < 50  # falling momentum
        
        if position == 0:
            if trending:
                if price_above_kama and rsi_bullish:
                    signals[i] = 0.25
                    position = 1
                elif price_below_kama and rsi_bearish:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below KAMA or RSI overbought
                if close[i] < kama[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above KAMA or RSI oversold
                if close[i] > kama[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals