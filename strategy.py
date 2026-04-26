#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v2
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index regime filter to avoid whipsaws.
KAMA adapts to market noise, reducing false signals in choppy conditions. RSI confirms momentum strength.
Chop filter ensures we only trade when market is trending (CHOP < 38.2) or mean-reverting (CHOP > 61.8) appropriately.
Designed for 30-100 total trades over 4 years (7-25/year) with discrete position sizing (0.0, ±0.25) to minimize fee churn.
Works in both bull and bear markets by adapting to regime and using symmetrical long/short logic.
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Load 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on 1w for higher timeframe trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA(10, 2, 30) on daily close
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Pad arrays to align with close
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.concatenate([[np.nan]*10, volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants: fastest = 2/(2+1)=0.6667, slowest = 2/(30+1)=0.0645
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9 (10th element)
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to align with close (first 14 values are NaN)
    rsi = np.concatenate([[np.nan]*14, rsi[:len(close)-14]])
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(1) over 14) / (max(high) - min(low) over 14)) / log10(14)
    atr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr1 = np.concatenate([[np.nan], atr1])  # Align with index 0
    sum_atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high14 - min_low14) > 0, chop, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(14, 20, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Regime filters
        is_trending = chop[i] < 38.2  # Trending market
        is_ranging = chop[i] > 61.8   # Ranging market
        
        # Long logic: 
        # In trending market: price > KAMA (uptrend) + RSI > 50 (bullish momentum) + 1w EMA20 uptrend
        # In ranging market: price < KAMA (mean reversion down) + RSI < 30 (oversold) + 1w EMA20 not strongly down
        if is_trending:
            long_condition = (close[i] > kama[i]) and (rsi[i] > 50) and (close[i] > ema_20_1w_aligned[i])
        else:  # ranging or choppy
            long_condition = (close[i] < kama[i]) and (rsi[i] < 30) and (close[i] > ema_20_1w_aligned[i] * 0.98)
        
        # Short logic:
        # In trending market: price < KAMA (downtrend) + RSI < 50 (bearish momentum) + 1w EMA20 downtrend
        # In ranging market: price > KAMA (mean reversion up) + RSI > 70 (overbought) + 1w EMA20 not strongly up
        if is_trending:
            short_condition = (close[i] < kama[i]) and (rsi[i] < 50) and (close[i] < ema_20_1w_aligned[i])
        else:  # ranging or choppy
            short_condition = (close[i] > kama[i]) and (rsi[i] > 70) and (close[i] < ema_20_1w_aligned[i] * 1.02)
        
        # Entry logic
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        # ATR-based stoploss: exit if price moves against position by 2.5 * ATR
        elif position == 1 and close[i] < kama[i] - 2.5 * atr[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > kama[i] + 2.5 * atr[i]:
            signals[i] = 0.0
            position = 0
        # Exit when KAMA cross in opposite direction (primary exit)
        elif position == 1 and close[i] < kama[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > kama[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0