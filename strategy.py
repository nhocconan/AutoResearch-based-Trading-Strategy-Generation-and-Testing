#!/usr/bin/env python3
# 6h_RSI_4hTrend_1dVolatilityBreakout_v1
# Hypothesis: Combines 6h RSI momentum with 4h trend filter and 1d volatility breakout.
# RSI(14) > 60 for long, < 40 for short, only when 4h EMA(50) confirms trend.
# Entry requires 1d ATR expansion (current ATR > 1.5x 10-day average) to avoid low-volatility whipsaws.
# Designed for 6h timeframe to capture medium-term moves in both bull and bear markets.
# Targets 15-25 trades/year to minimize fee drag. Position size 0.25.

name = "6h_RSI_4hTrend_1dVolatilityBreakout_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volatility breakout filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate RSI on 6h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR for volatility breakout
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr1.iloc[0] = 0
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_ma_10d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_expansion = atr_1d > (atr_ma_10d * 1.5)
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1d, atr_expansion)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 10)  # Warmup for 4h EMA, RSI, and ATR
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_expansion_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volatility breakout from 1d
        vol_breakout = atr_expansion_aligned[i]
        
        if position == 0:
            # Long entry: RSI > 60, 4h uptrend, and 1d volatility expansion
            if rsi[i] > 60 and uptrend and vol_breakout:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI < 40, 4h downtrend, and 1d volatility expansion
            elif rsi[i] < 40 and downtrend and vol_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI < 50 or trend turns down
            if rsi[i] < 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI > 50 or trend turns up
            if rsi[i] > 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals