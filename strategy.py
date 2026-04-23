#!/usr/bin/env python3
"""
Hypothesis: 12-hour RSI mean reversion with 1-day Bollinger Band squeeze and volume confirmation.
Long when RSI < 30, BB width < 0.05 (squeeze), and volume > 1.5x average.
Short when RSI > 70, BB width < 0.05 (squeeze), and volume > 1.5x average.
Exit when RSI returns to 50 or BB width expands > 0.10.
Designed for low trade frequency (~15-30/year) to capture mean reversion during low volatility.
Works in both bull and bear markets by exploiting oversold/overbought conditions during consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    
    # Calculate 12-hour RSI (14-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # Align HTF indicators to lower timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(bb_width_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_width_val = bb_width_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: RSI oversold, BB squeeze, volume confirmation
            if (rsi_val < 30 and bb_width_val < 0.05 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought, BB squeeze, volume confirmation
            elif (rsi_val > 70 and bb_width_val < 0.05 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI returns to neutral OR BB expands
                if rsi_val >= 50 or bb_width_val > 0.10:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI returns to neutral OR BB expands
                if rsi_val <= 50 or bb_width_val > 0.10:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_RSI_1dBB_Squeeze_Volume"
timeframe = "12h"
leverage = 1.0