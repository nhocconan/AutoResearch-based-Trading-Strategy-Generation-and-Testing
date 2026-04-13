#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h RSI(14) and volume confirmation.
# Long: RSI(12h) crosses above 60 + volume > 1.5x average volume (20-period).
# Short: RSI(12h) crosses below 40 + volume > 1.5x average volume.
# Uses 12h RSI for momentum filter, 6h for execution with volume confirmation.
# Volume confirmation reduces false breakouts and whipsaws.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate RSI(14) on 12h data
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_12h), np.nan)
    avg_loss = np.full(len(close_12h), np.nan)
    
    # Initialize averages
    if len(close_12h) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        
        # Wilder smoothing
        for i in range(14, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 12h RSI to 6h
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi = rsi_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: RSI crosses above 60 + volume confirmation
            if (rsi > 60 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: RSI crosses below 40 + volume confirmation
            elif (rsi < 40 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses below 50
            if rsi < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses above 50
            if rsi > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_RSI_Volume"
timeframe = "6h"
leverage = 1.0