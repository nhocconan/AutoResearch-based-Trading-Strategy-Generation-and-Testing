#!/usr/bin/env python3
"""
4h_MACD_Signal_Crossover_RSI_Filter_Volume
Hypothesis: Uses MACD line crossing signal line on 4h timeframe, filtered by RSI(50) for trend alignment and volume confirmation (>1.5x 20-period average). 
Designed to capture medium-term momentum with low trade frequency (~20-30 trades/year) to minimize fee drag. 
Works in both bull and bear markets by following momentum direction confirmed by RSI and volume.
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
    
    # MACD calculation on close prices
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for MACD, RSI, volume
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        macd_val = macd_line[i]
        signal_val = signal_line[i]
        rsi_val = rsi[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: MACD crosses above signal line, RSI > 50 (uptrend), volume confirmation
            if macd_val > signal_val and rsi_val > 50 and vol_conf:
                signals[i] = size
                position = 1
            # Short: MACD crosses below signal line, RSI < 50 (downtrend), volume confirmation
            elif macd_val < signal_val and rsi_val < 50 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: MACD crosses below signal line
            if macd_val < signal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: MACD crosses above signal line
            if macd_val > signal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_MACD_Signal_Crossover_RSI_Filter_Volume"
timeframe = "4h"
leverage = 1.0