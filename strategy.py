#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Williams %R mean reversion + volume confirmation + 12h EMA34 trend filter.
Long when Williams %R(14) crosses above -80 (oversold) with volume confirmation and price > 12h EMA34 (uptrend).
Short when Williams %R(14) crosses below -20 (overbought) with volume confirmation and price < 12h EMA34 (downtrend).
Exit when Williams %R returns to -50 (mean) or reverses with volume.
Williams %R identifies extreme momentum exhaustion, effective in both trending and ranging markets.
Volume confirmation filters low-conviction moves. 12h EMA34 provides intermediate trend filter.
Designed for 6h to balance trade frequency and capture multi-session reversals.
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
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d and 12h indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Williams %R signals: cross above -80 (long), cross below -20 (short)
        # Use previous bar to detect cross
        prev_williams = williams_r_aligned[i-1] if i > 0 else williams_r_aligned[i]
        curr_williams = williams_r_aligned[i]
        
        williams_long_signal = prev_williams <= -80 and curr_williams > -80
        williams_short_signal = prev_williams >= -20 and curr_williams < -20
        williams_exit_signal = (prev_williams < -50 and curr_williams >= -50) or \
                               (prev_williams > -50 and curr_williams <= -50)
        
        if position == 0:
            # Long: Williams %R crosses above -80 with volume and uptrend (price > EMA34)
            if (williams_long_signal and 
                volume_confirmed and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 with volume and downtrend (price < EMA34)
            elif (williams_short_signal and 
                  volume_confirmed and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR reverses with volume
            if (williams_exit_signal or 
                (curr_williams < -20 and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR reverses with volume
            if (williams_exit_signal or 
                (curr_williams > -80 and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dWilliamsR_MeanReversion_Volume_EMA34_Trend"
timeframe = "6h"
leverage = 1.0