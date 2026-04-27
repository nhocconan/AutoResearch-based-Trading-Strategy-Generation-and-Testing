#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filter.
Long when: price > KAMA, RSI > 50, CHOP > 61.8 (ranging market -> mean reversion to upside)
Short when: price < KAMA, RSI < 50, CHOP > 61.8 (ranging market -> mean reversion to downside)
Exit when trend reverses (price crosses KAMA) or CHOP < 38.2 (trending regime -> follow trend with KAMA).
Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades in strong trends.
Designed for low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drag.
Uses discrete position sizing (0.25) to reduce churn. Works in bull/bear markets: 
In ranging markets (CHOP > 61.8), mean reversion at KAMA with RSI momentum captures swings.
In trending markets (CHOP < 38.2), follow 1w EMA50 trend with KAMA as dynamic support/resistance.
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
    
    # Get 1d data for KAMA, RSI, CHOP
    df_1d = get_htf_data(prices, '1d')
    
    # Kaufman Adaptive Moving Average (KAMA)
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.divide(change, vol, out=np.zeros_like(change, dtype=float), where=vol!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr_14 = pd.Series(np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close_1d[:-1]), np.abs(low[1:] - close_1d[:-1])))).rolling(window=14, min_periods=14).mean().values
    atr_14 = np.insert(atr_14, 0, np.nan)  # align with close_1d
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / np.where((max_high - min_low) != 0, (max_high - min_low), 1)) / np.log10(14)
    
    # Align 1d indicators to lower timeframe (but we are on 1d, so no alignment needed)
    # Since timeframe is 1d, we can use the 1d arrays directly
    kama_1d = kama
    rsi_1d = rsi
    chop_1d = chop
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 1d indicators (KAMA needs 10, RSI 14, CHOP 14) and 1w EMA50
    start_idx = max(50, 14)  # 1w EMA50 needs 50 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_1d[i]
        rsi_val = rsi_1d[i]
        chop_val = chop_1d[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for entry: KAMA cross with RSI momentum in ranging market (CHOP > 61.8)
            long_condition = (close_val > kama_val and 
                            rsi_val > 50 and 
                            chop_val > 61.8)
            short_condition = (close_val < kama_val and 
                             rsi_val < 50 and 
                             chop_val > 61.8)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: trend reversal (price < KAMA) OR trending regime (CHOP < 38.2) with price < EMA50
            if close_val < kama_val or (chop_val < 38.2 and close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend reversal (price > KAMA) OR trending regime (CHOP < 38.2) with price > EMA50
            if close_val > kama_val or (chop_val < 38.2 and close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop"
timeframe = "1d"
leverage = 1.0