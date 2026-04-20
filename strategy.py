#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Fibonacci_Pullback_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: EMA50 for trend direction ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d: 20-period high/low for Fibonacci levels ===
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Fibonacci retracement levels (0.382, 0.618)
    fib_range = high_20 - low_20
    fib_0382 = low_20 + 0.382 * fib_range
    fib_0618 = low_20 + 0.618 * fib_range
    
    # Align 1d indicators
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    fib_0382_aligned = align_htf_to_ltf(prices, df_1d, fib_0382)
    fib_0618_aligned = align_htf_to_ltf(prices, df_1d, fib_0618)
    
    # === 6h: ATR(14) for volatility ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_50_1d_aligned[i]
        high_val = high_20_aligned[i]
        low_val = low_20_aligned[i]
        fib_0382_val = fib_0382_aligned[i]
        fib_0618_val = fib_0618_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_low = prices['low'].iloc[i]
        current_high = prices['high'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(high_val) or np.isnan(low_val) or
            np.isnan(fib_0382_val) or np.isnan(fib_0618_val) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. Price above 1d EMA50 (uptrend)
            # 2. Price pulls back to 0.618 Fib level and shows rejection (low touches/below level)
            # 3. Current close rebounds above 0.382 Fib level (confirmation)
            if (current_close > ema_trend and
                current_low <= fib_0618_val and  # Pullback to/deep 0.618
                current_close > fib_0382_val):   # Confirmation above 0.382
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below 1d EMA50 (downtrend)
            # 2. Price pulls back to 0.382 Fib level and shows rejection (high touches/above level)
            # 3. Current close drops below 0.618 Fib level (confirmation)
            elif (current_close < ema_trend and
                  current_high >= fib_0382_val and  # Pullback to/deep 0.382
                  current_close < fib_0618_val):    # Confirmation below 0.618
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below 1d EMA50 (trend change)
            # 2. ATR-based stop loss
            # 3. Take profit at 2x risk
            if (current_close < ema_trend or
                current_close < entry_price - 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            elif current_close >= entry_price + 4.0 * current_atr:  # 2:1 reward/risk
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above 1d EMA50 (trend change)
            # 2. ATR-based stop loss
            # 3. Take profit at 2x risk
            if (current_close > ema_trend or
                current_close > entry_price + 2.0 * current_atr):
                signals[i] = 0.0
                position = 0
            elif current_close <= entry_price - 4.0 * current_atr:  # 2:1 reward/risk
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals