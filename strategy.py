#!/usr/bin/env python3
# 4h_ema200_rsi_21_volume
# Hypothesis: 4h EMA200 trend filter + RSI21 momentum + volume confirmation. Long when price > EMA200 and RSI21 crosses above 50 with volume > 1.5x average. Short when price < EMA200 and RSI21 crosses below 50 with volume > 1.5x average. Uses daily EMA200 for stronger trend filter to avoid whipsaws. Designed for 4h timeframe to capture medium-term trends in both bull and bear markets with low trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema200_rsi_21_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
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
    
    # Calculate RSI(21) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/21, adjust=False, min_periods=21).mean()
    avg_loss = loss.ewm(alpha=1/21, adjust=False, min_periods=21).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 210
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA200 OR RSI crosses below 50
            if (close[i] < ema_200_1d_aligned[i]) or (rsi_values[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA200 OR RSI crosses above 50
            if (close[i] > ema_200_1d_aligned[i]) or (rsi_values[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Entry conditions
            if (close[i] > ema_200_1d_aligned[i]) and (rsi_values[i] > 50) and volume_ok:
                # Check for RSI crossing above 50 (momentum confirmation)
                if i > start_idx and rsi_values[i-1] <= 50:
                    position = 1
                    signals[i] = 0.25
            elif (close[i] < ema_200_1d_aligned[i]) and (rsi_values[i] < 50) and volume_ok:
                # Check for RSI crossing below 50 (momentum confirmation)
                if i > start_idx and rsi_values[i-1] >= 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals