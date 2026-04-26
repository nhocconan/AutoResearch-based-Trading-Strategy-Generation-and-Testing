#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI(14) for momentum confirmation and Choppiness Index(14) for regime filtering. Enter long when KAMA is rising, RSI > 50, and CHOP < 38.2 (trending regime). Enter short when KAMA is falling, RSI < 50, and CHOP < 38.2. Uses discrete position size 0.25 to limit drawdown and reduce fee churn. Designed for 7-25 trades/year on 1d by requiring strong trend alignment and momentum confirmation, avoiding whipsaws in ranging markets. Uses 1w EMA50 as higher timeframe trend filter to ensure alignment with weekly structure.
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
    
    # Get 1d data for KAMA, RSI, Chop and 1w for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA(10) on 1d close
    close_1d = pd.Series(df_1d['close'].values)
    # Efficiency Ratio
    change = abs(close_1d.diff(10).values)
    volatility = abs(close_1d.diff(1)).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d.iloc[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d.iloc[i] - kama[i-1])
    kama_values = kama
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    
    # Calculate RSI(14) on 1d close
    delta = close_1d.diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index(14) on 1d high/low/close
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1].values)
    tr3 = np.abs(low[1:] - close_1d[:-1].values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close index
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of True Range over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # MaxHigh - MinLow over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    # Choppiness Index
    chop = np.where(range_hl > 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA seed, RSI, Chop, and EMA warmup
    start_idx = max(30, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA direction (using 1-bar change)
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI momentum
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        
        # Chop regime filter: trending market
        chop_trending = chop_aligned[i] < 38.2
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, Chop < 38.2, 1w uptrend
            long_signal = kama_rising and rsi_above_50 and chop_trending and trend_1w_uptrend
            
            # Short: KAMA falling, RSI < 50, Chop < 38.2, 1w downtrend
            short_signal = kama_falling and rsi_below_50 and chop_trending and trend_1w_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA falling OR Chop > 38.2 (ranging) OR 1w trend turns down
            if (not kama_rising or chop_aligned[i] >= 38.2 or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA rising OR Chop > 38.2 (ranging) OR 1w trend turns up
            if (not kama_falling or chop_aligned[i] >= 38.2 or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0