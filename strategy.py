#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI for momentum and Choppiness Index for regime filtering. Long when KAMA slopes up,
RSI > 50, and market is trending (CHOP < 38.2). Short when KAMA slopes down, RSI < 50, and CHOP < 38.2.
Avoids ranging markets (CHOP > 61.8) to reduce false signals. Designed for low trade frequency and robustness in both bull and bear markets.
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
    
    # Calculate KAMA (primary trend) on 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # KAMA parameters: ER period=10, fastest EMA=2, slowest EMA=30
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period sum of absolute changes
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility  # Efficiency Ratio
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Smoothing Constant
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to original timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI (14) on 1d data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    # Align RSI to original timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index (14) on 1d data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero and log of zero
    hh_ll = hh - ll
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(14)
    # Align Chop to original timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate KAMA slope (1-period change)
        if i > 0 and not np.isnan(kama_aligned[i-1]):
            kama_slope = kama_aligned[i] - kama_aligned[i-1]
        else:
            kama_slope = 0
        
        ema_trend = ema_50_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            if is_trending:
                # Long: KAMA rising, RSI > 50, and price above weekly EMA50 (uptrend bias)
                if kama_slope > 0 and rsi_val > 50 and close[i] > ema_trend:
                    signals[i] = 0.25
                    position = 1
                # Short: KAMA falling, RSI < 50, and price below weekly EMA50 (downtrend bias)
                elif kama_slope < 0 and rsi_val < 50 and close[i] < ema_trend:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # In ranging market, stay flat
                signals[i] = 0.0
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
            # Exit: KAMA slope turns negative OR RSI < 40 OR chop > 61.8 (ranging) OR trend reversal
            if (kama_slope < 0) or (rsi_val < 40) or (chop_val > 61.8) or (close[i] < ema_trend * 0.99):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short position
            signals[i] = -0.25
            # Exit: KAMA slope turns positive OR RSI > 60 OR chop > 61.8 (ranging) OR trend reversal
            if (kama_slope > 0) or (rsi_val > 60) or (chop_val > 61.8) or (close[i] > ema_trend * 1.01):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0