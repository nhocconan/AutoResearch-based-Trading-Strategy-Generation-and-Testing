#!/usr/bin/env python3
# 1d_1w_ema_crossover_volatility_breakout
# Strategy: Daily EMA crossover with weekly trend filter and volatility breakout
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: EMA(9/21) crossovers signal trend changes. Weekly EMA(50) filters trend direction.
# Volatility breakout (ATR-based) confirms momentum. Works in bull by riding uptrends,
# and in bear by catching short-term reversals against the weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_crossover_volatility_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_trend = ema_50_1w > np.roll(ema_50_1w, 1)  # Rising EMA = uptrend
    ema_50_1w_trend[0] = False  # First value invalid
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_trend)
    
    # Daily EMA(9) and EMA(21) for crossover
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR(14) for volatility breakout
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Signals
    bullish_cross = ema_9 > ema_21
    bearish_cross = ema_9 < ema_21
    
    # Volatility breakout: price moves beyond 1.5 * ATR from prior close
    bullish_break = close > np.roll(close, 1) + 1.5 * atr
    bearish_break = close < np.roll(close, 1) - 1.5 * atr
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: EMA crossover + volatility breakout + weekly trend alignment
        bullish_entry = (bullish_cross[i] and bullish_break[i] and ema_50_1w_aligned[i])
        bearish_entry = (bearish_cross[i] and bearish_break[i] and not ema_50_1w_aligned[i])
        
        # Exit conditions: opposing EMA crossover
        exit_long = position == 1 and bearish_cross[i]
        exit_short = position == -1 and bullish_cross[i]
        
        # Trading logic
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
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