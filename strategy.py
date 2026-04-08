#!/usr/bin/env python3
# 4h_rsi_ema_crossover_1d_trend_volume
# Hypothesis: RSI(14) crossing above/below EMA(21) on 4h with 1d EMA(100) trend filter and volume confirmation.
# Long when RSI crosses above EMA(21), price > 1d EMA(100), and volume > 1.5x average.
# Short when RSI crosses below EMA(21), price < 1d EMA(100), and volume > 1.5x average.
# Exit when RSI crosses back over EMA(21) in opposite direction.
# Designed to capture momentum with trend alignment in both bull and bear markets.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_ema_crossover_1d_trend_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA(21) on 4h
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below EMA(21)
            if rsi[i] < ema_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above EMA(21)
            if rsi[i] > ema_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # RSI crossing EMA signals
            rsi_cross_above = rsi[i] > ema_21[i] and rsi[i-1] <= ema_21[i-1]
            rsi_cross_below = rsi[i] < ema_21[i] and rsi[i-1] >= ema_21[i-1]
            
            # Entry conditions with trend filter
            if rsi_cross_above and (close[i] > ema_100_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif rsi_cross_below and (close[i] < ema_100_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals