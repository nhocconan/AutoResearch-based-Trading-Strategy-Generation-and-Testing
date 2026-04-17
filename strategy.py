#!/usr/bin/env python3
"""
4h_TrueRange_Reversal_With_Trend_Filter
Hypothesis: Use Average True Range (ATR) to detect volatility spikes that signal potential reversals.
Combine with daily trend filter (price > EMA50) for long bias in uptrends and short bias in downtrends.
ATR-based entries help capture volatility expansion moves, which are common at trend reversals.
Designed for low trade frequency (<25/year) to minimize fee drag and improve robustness across market regimes.
"""

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
    
    # === 4x True Range and ATR(14) ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h EMA(50) for trend filter ===
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === Daily ATR for volatility regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values  # 10-day ATR average
    atr_1d_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # === Daily ATR spike detection: current ATR > 1.5x 10-day avg ATR ===
    atr_1d_current = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_spike = atr_1d_current > 1.5 * atr_1d_avg_aligned
    
    signals = np.zeros(n)
    
    # Warmup: covers ATR(14), EMA(50), and 10-day ATR average
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr[i]) or np.isnan(ema50[i]) or 
            np.isnan(atr_spike[i]) or np.isnan(atr_1d_current[i]) or np.isnan(atr_1d_avg_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: ATR spike + price above EMA50 (uptrend bias)
            if atr_spike[i] and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: ATR spike + price below EMA50 (downtrend bias)
            elif atr_spike[i] and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: volatility contraction (ATR drops below 0.8x 4x ATR average)
        elif position == 1:
            if atr[i] < 0.8 * np.nanmean(atr[max(0, i-20):i+1]):  # volatility contraction
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if atr[i] < 0.8 * np.nanmean(atr[max(0, i-20):i+1]):  # volatility contraction
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TrueRange_Reversal_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0