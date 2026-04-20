#!/usr/bin/env python3
"""
6h_1w_1d_RSI_Divergence_Momentum
Concept: 6s RSI divergence with weekly trend and daily momentum confirmation.
- Long: Weekly EMA(20) > EMA(50) AND daily RSI(14) < 30 (oversold) AND 6h RSI(14) > previous 6h RSI(14) (bullish divergence)
- Short: Weekly EMA(20) < EMA(50) AND daily RSI(14) > 70 (overbought) AND 6h RSI(14) < previous 6h RSI(14) (bearish divergence)
- Exit: RSI crosses back to neutral (40-60 range) or weekly trend flips
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years
- Works in bull/bear: Weekly trend filter adapts, RSI divergence captures exhaustion points
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_RSI_Divergence_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Weekly: EMA Trend Filter (20 and 50) ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMAs to 6h
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily: RSI (14) ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # === 6h: RSI (14) for divergence detection ===
    close = prices['close'].values
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_6h = 100 - (100 / (1 + rs))
    rsi_6h_values = rsi_6h.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema20 = ema_20_1w_aligned[i]
        ema50 = ema_50_1w_aligned[i]
        rsi_daily = rsi_1d_aligned[i]
        rsi_current = rsi_6h_values[i]
        rsi_previous = rsi_6h_values[i-1]
        
        # Skip if any value is NaN
        if (np.isnan(ema20) or np.isnan(ema50) or np.isnan(rsi_daily) or 
            np.isnan(rsi_current) or np.isnan(rsi_previous)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend + daily oversold + bullish RSI divergence
            if ema20 > ema50 and rsi_daily < 30 and rsi_current > rsi_previous:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + daily overbought + bearish RSI divergence
            elif ema20 < ema50 and rsi_daily > 70 and rsi_current < rsi_previous:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or weekly trend flips
            if rsi_current > 40 or ema20 < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or weekly trend flips
            if rsi_current < 60 or ema20 > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals