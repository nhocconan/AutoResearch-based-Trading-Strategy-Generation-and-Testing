#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies trend direction with lag reduction.
RSI(14) filters overbought/oversold conditions within the trend.
Choppiness Index (CHOP) > 61.8 defines ranging regime where mean reversion works.
Only take long when: KAMA up, RSI < 40, CHOP > 61.8 (pullback in uptrend within range).
Only take short when: KAMA down, RSI > 60, CHOP > 61.8 (bounce in downtrend within range).
Uses 1w EMA200 as trend filter to avoid counter-trend trades in strong trends.
Designed for 1d timeframe with tight entry conditions to achieve 7-25 trades/year.
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
    
    # Get 1d data for indicators (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(df_1d['close'].values, n=10))
    volatility = np.sum(np.abs(np.diff(df_1d['close'].values)), axis=1)
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(df_1d['close'].values, np.nan)
    kama[9] = df_1d['close'].values[9]  # Start after 10 periods
    for i in range(10, len(kama)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (df_1d['close'].values[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 1d timeframe (already 1d, but use align for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index (CHOP) on 1d
    # True Range
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr3 = np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Max/min close over 14 periods
    max_close = pd.Series(df_1d['close'].values).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(df_1d['close'].values).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop = np.where(
        (atr * 14) > 0,
        100 * np.log10((max_close - min_close) / (atr * 14)) / np.log10(14),
        50  # neutral when no volatility
    )
    # Pad beginning with NaN
    chop = np.concatenate([np.full(13, np.nan), chop])
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate EMA200 on 1w close for trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_200_val = ema_200_1w_aligned[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals
            # Long: KAMA up (trend up), RSI < 40 (oversold), in range
            long_entry = (curr_close > kama_val) and (rsi_val < 40) and in_range
            # Short: KAMA down (trend down), RSI > 60 (overbought), in range
            short_entry = (curr_close < kama_val) and (rsi_val > 60) and in_range
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: KAMA down (trend change) OR RSI > 70 (overbought) OR chop < 38.2 (trending regime)
            if (curr_close < kama_val) or (rsi_val > 70) or (chop_val < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: KAMA up (trend change) OR RSI < 30 (oversold) OR chop < 38.2 (trending regime)
            if (curr_close > kama_val) or (rsi_val < 30) or (chop_val < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0