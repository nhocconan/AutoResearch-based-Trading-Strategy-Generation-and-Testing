#!/usr/bin/env python3
"""
12h_1d_volatility_breakout
Strategy: 12h volatility breakout with 1-day volatility filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses 12-hour volatility expansion (ATR > 1.5x 20-period average) combined with 1-day low volatility regime (ATR ratio < 0.6) to capture breakouts during volatility expansion phases. Works in both bull and bear markets by trading volatility expansion/contraction cycles. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_volatility_breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h ATR for volatility breakout signal
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_12h = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    atr_ratio_12h = atr_12h / atr_ma_20_12h
    
    # === 1-day ATR (volatility filter: low volatility regime) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1-day ATR ratio: current ATR / 20-period average ATR (low when < 0.6)
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_20_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(atr_ratio_12h[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_open = open_price[i]
        
        # Volatility breakout: 12h ATR expansion > 1.5x 20-period average
        volatility_expansion = atr_ratio_12h[i] > 1.5
        
        # Volatility filter: low volatility regime (1-day ATR ratio < 0.6)
        low_volatility = atr_ratio_1d_aligned[i] < 0.6
        
        # Strong candle: close > open for longs, close < open for shorts
        strong_bullish = price_close > price_open
        strong_bearish = price_close < price_open
        
        # Long conditions: volatility expansion + low volatility regime + strong bullish candle
        long_signal = volatility_expansion and low_volatility and strong_bullish
        
        # Short conditions: volatility expansion + low volatility regime + strong bearish candle
        short_signal = volatility_expansion and low_volatility and strong_bearish
        
        # Exit when volatility contraction occurs (ATR ratio < 1.2)
        exit_long = position == 1 and atr_ratio_12h[i] < 1.2
        exit_short = position == -1 and atr_ratio_12h[i] < 1.2
        
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

# Hypothesis: Uses 12-hour volatility expansion (ATR > 1.5x 20-period average) combined with 1-day low volatility regime (ATR ratio < 0.6) to capture breakouts during volatility expansion phases. Works in both bull and bear markets by trading volatility expansion/contraction cycles. Target: 15-25 trades/year.