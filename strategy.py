#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_SMMA_Fractal_Momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === SMMA (Smoothed Moving Average) - 1d ===
    close_1d = df_1d['close'].values
    smma_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    if len(close_1d) >= 10:
        smma_1d[9] = np.mean(close_1d[:10])
        for i in range(10, len(close_1d)):
            smma_1d[i] = (smma_1d[i-1] * 9 + close_1d[i]) / 10
    smma_1d_aligned = align_htf_to_ltf(prices, df_1d, smma_1d)
    
    # === Williams Fractals - 1d (requires extra delay for confirmation) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate fractals
    bearish_fractal = np.zeros_like(high_1d)
    bullish_fractal = np.zeros_like(low_1d)
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Apply extra delay for fractal confirmation (2 bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # === 6h Volume filter ===
    vol_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # === 6h RSI for momentum confirmation ===
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for SMMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(smma_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma10[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above SMMA + bullish fractal + RSI > 50 + volume confirmation
            long_cond = (close[i] > smma_1d_aligned[i] and 
                        bullish_fractal_aligned[i] > 0 and
                        rsi[i] > 50 and
                        volume[i] > vol_ma10[i])
            
            # Short: price below SMMA + bearish fractal + RSI < 50 + volume confirmation
            short_cond = (close[i] < smma_1d_aligned[i] and 
                         bearish_fractal_aligned[i] > 0 and
                         rsi[i] < 50 and
                         volume[i] > vol_ma10[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below SMMA or bearish fractal or RSI < 40
            exit_cond = (close[i] < smma_1d_aligned[i] or 
                        bearish_fractal_aligned[i] > 0 or
                        rsi[i] < 40)
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above SMMA or bullish fractal or RSI > 60
            exit_cond = (close[i] > smma_1d_aligned[i] or 
                        bullish_fractal_aligned[i] > 0 or
                        rsi[i] > 60)
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines SMMA (smooth trend) with Williams Fractals (key reversal points) on daily timeframe.
# SMMA provides adaptive trend filter, fractals identify key reversal levels for entries.
# Volume and RSI add confirmation. Designed for 6h timeframe to capture multi-day moves.
# Works in bull markets via trend following (price > SMMA) and in bear markets via 
# mean reversion at fractal levels. Targets 50-100 trades over 4 years to minimize fee drag.