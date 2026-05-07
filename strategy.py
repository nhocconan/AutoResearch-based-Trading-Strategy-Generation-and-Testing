#!/usr/bin/env python3
"""
4H_Momentum_Volume_Squeeze_1D_Trend_v1
Hypothesis: Combine 4h momentum (RSI > 50 for long, < 50 for short) with 1d trend (close > EMA50 for long, < EMA50 for short) and volume spikes (>1.5x average). 
Enter long when momentum is bullish, trend is bullish, and volume spikes; short when momentum is bearish, trend is bearish, and volume spikes. 
Exit when momentum reverses. Uses volume confirmation to filter false signals and trend alignment to work in both bull and bear markets.
"""
name = "4H_Momentum_Volume_Squeeze_1D_Trend_v1"
timeframe = "4h"
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
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = pd.Series(df_1d['close'])
    ema_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (1.3 days on 4h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Long: RSI > 50 (bullish momentum), close > EMA50 (bullish trend), volume spike
            if (rsi[i] > 50 and close[i] > ema_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: RSI < 50 (bearish momentum), close < EMA50 (bearish trend), volume spike
            elif (rsi[i] < 50 and close[i] < ema_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: momentum reverses (RSI crosses 50)
            if position == 1 and rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals