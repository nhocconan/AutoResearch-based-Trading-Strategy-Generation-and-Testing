#!/usr/bin/env python3
# 6h_ema200_rsi14_volume
# Hypothesis: Trend-following with EMA200 filter on 1d timeframe and RSI(14) momentum confirmation on 6h.
# Long when price > EMA200(1d) and RSI(14) crosses above 50 with volume > 1.3x average.
# Short when price < EMA200(1d) and RSI(14) crosses below 50 with volume > 1.3x average.
# Designed to capture trend continuations in both bull and bear markets with proper filtering.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema200_rsi14_volume"
timeframe = "6h"
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
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate RSI(14) on 6h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI falls below 40 OR price crosses below EMA200
            if (rsi[i] < 40) or (close[i] < ema_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI rises above 60 OR price crosses above EMA200
            if (rsi[i] > 60) or (close[i] > ema_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # Entry conditions: RSI crossing 50 with trend and volume confirmation
            if (rsi[i] > 50) and (rsi[i-1] <= 50) and (close[i] > ema_200_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (rsi[i] < 50) and (rsi[i-1] >= 50) and (close[i] < ema_200_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals