#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, KAMA(10,2,30) trend direction + RSI(14) extreme + Choppiness Index(14) regime filter produces high-quality, low-frequency trades. KAMA adapts to market noise, RSI avoids overextended entries, and chop filter ensures mean-reversion only in ranging markets (CHOP > 61.8). Targets 7-25 trades/year with discrete position sizing (0.0, ±0.30) to minimize fee drag. Works in both bull and bear markets by adapting to regime.
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
    
    # Load 1w data ONCE before loop for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for higher timeframe trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # KAMA(10,2,30) - Adaptive trend indicator
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # (ER * (fast - slow) + slow)^2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to align with close (first 14 values are NaN)
    rsi_padded = np.full_like(close, np.nan, dtype=float)
    rsi_padded[14:] = rsi
    
    # Choppiness Index(14) - measures ranging vs trending
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to 0 (no previous close)
    tr[0] = 0
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero and log of zero
    range_hl = hh - ll
    chop = np.where((range_hl > 0) & (atr_sum > 0), 
                    100 * np.log10(atr_sum / range_hl) / np.log10(14), 
                    50)  # default to 50 (neutral) if invalid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Warmup: max of KAMA seed (10), RSI (14), Chop (14), EMA50_1w (50)
    start_idx = max(10, 14, 14, 50)
    
    for i in range(start_idx, n):
        kama_val = kama[i]
        close_val = close[i]
        rsi_val = rsi_padded[i]
        chop_val = chop[i]
        trend_val = ema50_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or np.isnan(trend_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # KAMA direction: price > KAMA = uptrend, price < KAMA = downtrend
        is_uptrend = close_val > kama_val
        is_downtrend = close_val < kama_val
        
        # RSI extremes: < 30 = oversold, > 70 = overbought
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
        is_ranging = chop_val > 61.8
        
        # Entry conditions: 
        # Long: KAMA uptrend + RSI oversold + ranging market (mean reversion in uptrend)
        # Short: KAMA downtrend + RSI overbought + ranging market (mean reversion in downtrend)
        long_entry = is_uptrend and rsi_oversold and is_ranging
        short_entry = is_downtrend and rsi_overbought and is_ranging
        
        # Exit conditions: 
        # Long exit: RSI > 50 (momentum shift) OR Chop < 38.2 (trending regime) OR price < KAMA (trend break)
        # Short exit: RSI < 50 OR Chop < 38.2 OR price > KAMA
        long_exit = (rsi_val > 50) or (chop_val < 38.2) or (close_val < kama_val)
        short_exit = (rsi_val < 50) or (chop_val < 38.2) or (close_val > kama_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
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

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0