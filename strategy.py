#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_And_Chop_Filter
Hypothesis: Daily KAMA trend direction combined with RSI extremes and Choppiness Index regime filter.
KAMA adapts to market noise, reducing whipsaw in sideways markets. RSI < 30 for long, > 70 for short in trending regimes.
Chop > 61.8 = range (avoid trend trades), Chop < 38.2 = trend (allow trend trades).
Designed for BTC/ETH robustness in both bull and bear markets via adaptive trend filter and regime avoidance.
Targets 30-100 trades over 4 years (7-25/year) with 0.25 position size. Uses discrete levels to minimize fee drag.
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
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA on daily close
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    er[10:] = change / np.where(volatility[10:] == 0, 1, volatility[10:])
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi_padded = np.concatenate([np.full(14, np.nan), rsi])
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_padded)
    
    # Calculate Choppiness Index(14) on 1d
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Max/min close over 14 periods
    max_close = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(atr14) / (max_close - min_close)) / log10(14)
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    denominator = max_close - min_close
    # Avoid division by zero and log of zero
    chop = np.zeros_like(close_1d)
    mask = (denominator > 0) & (~np.isnan(denominator))
    chop[mask] = 100 * np.log10(sum_atr14[mask] / denominator[mask]) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # Align with close (14 periods needed)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 1w EMA34 (34), KAMA (10+14=24), RSI (14), Chop (13+14=27)
    start_idx = max(34 + 1, 24 + 1, 14 + 1, 27 + 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        ema_val = ema_34_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only allow trend trades when Chop < 38.2
        regime_allow = chop_val < 38.2
        
        if position == 0:
            # Look for entry: KAMA trend alignment + RSI extreme + regime filter
            long_condition = (close_val > kama_val and 
                            close_val > ema_val and 
                            rsi_val < 30 and 
                            regime_allow)
            short_condition = (close_val < kama_val and 
                             close_val < ema_val and 
                             rsi_val > 70 and 
                             regime_allow)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA (trend reversal) or RSI > 70 (overbought)
            if close_val < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA (trend reversal) or RSI < 30 (oversold)
            if close_val > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_RSI_And_Chop_Filter"
timeframe = "1d"
leverage = 1.0