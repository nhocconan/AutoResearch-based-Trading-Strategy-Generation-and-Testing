#!/usr/bin/env python3
name = "6h_Choppiness_Adaptive_Strategy"
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
    
    # Get 1d data for regime detection (Choppiness Index)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) and min(low) over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_atr_14 / (max_high_14 - min_low_14)) / log10(14)
    range_14 = max_high_14 - min_low_14
    choppy = np.where(
        (range_14 > 0) & (~np.isnan(sum_atr_14)) & (~np.isnan(range_14)),
        100 * np.log10(sum_atr_14 / range_14) / np.log10(14),
        50  # Default to neutral when invalid
    )
    
    # Align Choppiness to 6h timeframe
    choppy_aligned = align_htf_to_ltf(prices, df_1d, choppy)
    
    # Get 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6-period RSI for entry timing
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=6, min_periods=6).mean().values
    avg_loss = pd.Series(loss).rolling(window=6, min_periods=6).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first value with NaN
    rsi = np.concatenate([[np.nan], rsi])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(choppy_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        choppy_val = choppy_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # In trending regime (Choppiness < 38.2): follow trend with RSI pullback
            if choppy_val < 38.2:
                # Uptrend: price above EMA50, RSI pulling back from overbought
                if close[i] > ema_trend and 40 < rsi_val < 50:
                    signals[i] = 0.25
                    position = 1
                # Downtrend: price below EMA50, RSI bouncing from oversold
                elif close[i] < ema_trend and 50 < rsi_val < 60:
                    signals[i] = -0.25
                    position = -1
            # In ranging regime (Choppiness > 61.8): mean reversion at extremes
            elif choppy_val > 61.8:
                # Buy near oversold RSI in range
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                # Sell near overbought RSI in range
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend weakening or overbought in range
            if choppy_val > 61.8 and rsi_val > 70:  # Range + overbought
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_trend:  # Trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakening or oversold in range
            if choppy_val > 61.8 and rsi_val < 30:  # Range + oversold
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_trend:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals