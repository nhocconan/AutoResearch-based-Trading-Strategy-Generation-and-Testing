#!/usr/bin/env python3
"""
4h_Adaptive_Trend_Strategy
Hypothesis: Combine adaptive trend detection with volume confirmation and regime filtering. Uses EMA crossover with ATR-based dynamic thresholds to capture trends while avoiding whipsaws. In bull markets: captures uptrends with minimal drawdown. In bear markets: avoids false breakdowns through volatility-adjusted filters. Targets 20-30 trades/year by requiring multiple confirmations (trend + volume + volatility regime).
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
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily ATR for volatility regime (14-period)
    atr_14 = pd.Series(np.maximum(
        df_1d['high'] - df_1d['low'],
        np.maximum(
            abs(df_1d['high'] - df_1d['close'].shift(1)),
            abs(df_1d['low'] - df_1d['close'].shift(1))
        )
    )).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 4-hour EMA crossover system (8 and 21)
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_threshold = vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 30, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        ema_fast = ema8[i]
        ema_slow = ema21[i]
        vol_ok = volume[i] > vol_threshold[i]
        
        # Dynamic threshold based on volatility
        dyn_threshold = atr_val * 0.5
        
        if position == 0:
            # Long: EMA8 > EMA21 + volume + price above EMA50 + bullish momentum
            if (ema_fast > ema_slow + dyn_threshold and 
                vol_ok and 
                close[i] > ema_trend and
                close[i] > close[i-1]):  # additional momentum filter
                signals[i] = size
                position = 1
            # Short: EMA8 < EMA21 - volume + price below EMA50 + bearish momentum
            elif (ema_fast < ema_slow - dyn_threshold and 
                  vol_ok and 
                  close[i] < ema_trend and
                  close[i] < close[i-1]):  # additional momentum filter
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: EMA crossover down OR volatility spike against trend
            if (ema_fast < ema_slow or 
                close[i] < ema_trend - atr_val or
                not vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: EMA crossover up OR volatility spike against trend
            if (ema_fast > ema_slow or 
                close[i] > ema_trend + atr_val or
                not vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Adaptive_Trend_Strategy"
timeframe = "4h"
leverage = 1.0