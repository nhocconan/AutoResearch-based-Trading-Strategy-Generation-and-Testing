#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering.
Only take longs when price > KAMA, RSI > 50, and CHOP < 61.8 (trending regime).
Only take shorts when price < KAMA, RSI < 50, and CHOP < 61.8.
In choppy regimes (CHOP >= 61.8), remain flat to avoid whipsaw.
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to limit drawdown and fee drag.
Designed to work in both bull and bear markets by adapting to trending regimes only.
Timeframe: 1d, uses 1w HTF for trend filter (EMA34 on weekly close).
Target: 30-100 total trades over 4 years = 7-25/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA34 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 60):
        return np.zeros(n)
    
    # === 1w OHLC for EMA34 trend filter ===
    df_1w_close = df_1w['close'].values
    ema_34_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === KAMA (10, 2, 30) on daily close ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else np.zeros_like(close)
    # Avoid division by zero
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) on daily close ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50.0
    
    # === Choppiness Index(14) ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where(atr_sum != 0, -100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14), 50)
    # Handle first 14 values
    chop[:14] = 50.0
    
    # === ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Only trade in trending regime (CHOP < 61.8)
        trending_regime = chop_val < 61.8
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, trending regime, and weekly trend alignment (price > weekly EMA34)
            long_condition = (price > kama_val) and (rsi_val > 50) and trending_regime and (price > ema_trend)
            # Short: price < KAMA, RSI < 50, trending regime, and weekly trend alignment (price < weekly EMA34)
            short_condition = (price < kama_val) and (rsi_val < 50) and trending_regime and (price < ema_trend)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < kama_val:
                signals[i] = 0.0
                position = 0
            # Regime change to choppy
            elif chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > kama_val:
                signals[i] = 0.0
                position = 0
            # Regime change to choppy
            elif chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0