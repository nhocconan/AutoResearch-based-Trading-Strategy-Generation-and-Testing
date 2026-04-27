#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA trend filter and volume confirmation
# Williams Fractals identify potential reversal points (Lows for long, Highs for short).
# Only take breakouts in direction of 1d EMA(34) trend with volume > 1.3x average.
# Works in both bull and bear markets by aligning with higher timeframe trend.
# Target: 15-30 trades/year to minimize fee decay while capturing high-probability reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    
    # Calculate EMA with proper initialization
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Calculate Williams Fractals on 6h data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        # Bearish fractal (peak)
        if (high[i-2] < high[i-1] and 
            high[i] < high[i-1] and 
            high[i-3] < high[i-1] and 
            high[i+1] < high[i-1]):
            bearish_fractal[i-1] = True
            
        # Bullish fractal (trough)
        if (low[i-2] > low[i-1] and 
            low[i] > low[i-1] and 
            low[i-3] > low[i-1] and 
            low[i+1] > low[i-1]):
            bullish_fractal[i-1] = True
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align HTF indicators to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(20, 34) + 2  # vol_period + fractal lookback + buffer
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. Fresh fractal formation (within last 3 bars to avoid stale signals)
        # 2. EMA trend filter: price > EMA for longs, price < EMA for shorts
        # 3. Volume confirmation: > 1.3x average volume
        fresh_bearish = bearish_fractal[i-2] or bearish_fractal[i-1] or bearish_fractal[i]
        fresh_bullish = bullish_fractal[i-2] or bullish_fractal[i-1] or bullish_fractal[i]
        trend_filter_long = price > ema_1d_aligned[i]
        trend_filter_short = price < ema_1d_aligned[i]
        volume_confirmation = vol_ratio > 1.3
        
        if position == 0:
            # Long: bullish fractal with uptrend and volume
            if fresh_bullish and trend_filter_long and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: bearish fractal with downtrend and volume
            elif fresh_bearish and trend_filter_short and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA or fractal failure
            if price < ema_1d_aligned[i] or fresh_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above EMA or fractal failure
            if price > ema_1d_aligned[i] or fresh_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_EMATrend_Volume"
timeframe = "6h"
leverage = 1.0