#!/usr/bin/env python3
"""
Hypothesis: 12h price reversal at 1d Bollinger Bands with volume confirmation and 1d RSI filter.
- Long: price touches/below lower Bollinger Band (20,2), RSI < 30, volume > 1.5x average
- Short: price touches/above upper Bollinger Band (20,2), RSI > 70, volume > 1.5x average
- Exit: price crosses middle Bollinger Band (20-day SMA)
- Uses Bollinger Bands for mean reversion in ranging markets, effective in both bull and bear.
Designed for 12-37 trades/year (50-150 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    if len(close) >= period:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if rs[i] is not np.nan:
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands."""
    if len(close) < period:
        return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)
    
    sma = np.full(len(close), np.nan)
    std = np.full(len(close), np.nan)
    
    for i in range(period - 1, len(close)):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std[i] = np.std(close[i - period + 1:i + 1])
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return upper, lower, sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20,2) on 1d
    upper_bb, lower_band, middle_bb = calculate_bollinger_bands(close_1d, 20, 2)
    
    # Calculate RSI (14) on 1d
    rsi_14_1d = calculate_rsi(close_1d, 14)
    
    # Align to 12h timeframe
    upper_bb_12h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_band_12h = align_htf_to_ltf(prices, df_1d, lower_band)
    middle_bb_12h = align_htf_to_ltf(prices, df_1d, middle_bb)
    rsi_14_1d_12h = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need Bollinger Bands, RSI, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_12h[i]) or np.isnan(lower_band_12h[i]) or 
            np.isnan(middle_bb_12h[i]) or np.isnan(rsi_14_1d_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price touches/below lower BB, RSI < 30, volume confirmation
            if close[i] <= lower_band_12h[i] and rsi_14_1d_12h[i] < 30 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price touches/above upper BB, RSI > 70, volume confirmation
            elif close[i] >= upper_bb_12h[i] and rsi_14_1d_12h[i] > 70 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above middle BB
            if close[i] >= middle_bb_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below middle BB
            if close[i] <= middle_bb_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Bollinger20_2_RSI14_Volume"
timeframe = "12h"
leverage = 1.0