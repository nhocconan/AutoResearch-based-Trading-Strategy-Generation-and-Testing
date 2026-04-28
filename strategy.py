#!/usr/bin/env python3
# Hypothesis: 4h RSI mean reversion with Bollinger Bands and volume confirmation in ranging markets.
# In ranging markets (low volatility), price tends to revert to the mean when hitting Bollinger Bands.
# RSI < 30 indicates oversold (long signal), RSI > 70 indicates overbought (short signal).
# Bollinger Band width < 0.05 identifies low volatility ranging conditions.
# Volume confirmation ensures reversals have participation.
# Works in both bull and bear markets by focusing on range-bound conditions.

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
    
    # Get daily data for Bollinger Bands and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_period = 20
    bb_std = 2
    
    # Middle band (SMA)
    bb_middle = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    # Standard deviation
    bb_std_dev = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    # Upper and lower bands
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    # Bollinger Band width (normalized)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width = np.where(bb_middle != 0, bb_width, 0)
    
    # Bollinger Band width < 0.05 indicates low volatility ranging market
    ranging_filter = bb_width < 0.05
    
    # Align ranging filter to 4h timeframe
    ranging_filter_aligned = align_htf_to_ltf(prices, df_1d, ranging_filter)
    
    # Calculate RSI (14-period) on 4h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Smoothed average gain/loss
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # RS and RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ranging_filter_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry conditions with volume confirmation
        long_entry = rsi_oversold and ranging_filter_aligned[i] and volume_filter[i]
        short_entry = rsi_overbought and ranging_filter_aligned[i] and volume_filter[i]
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        long_exit = rsi[i] >= 40
        short_exit = rsi[i] <= 60
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_MeanReversion_BBands_RangeFilter_Volume"
timeframe = "4h"
leverage = 1.0