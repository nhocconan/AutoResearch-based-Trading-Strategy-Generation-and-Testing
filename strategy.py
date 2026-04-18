#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily breakout of weekly Bollinger Bands with volume confirmation.
# Uses weekly Bollinger Bands (20, 2.0) to define volatility regime and trend.
# Enters on daily close beyond upper/lower band with volume > 1.5x 20-day average.
# Exits on return to middle band (20-day SMA).
# Designed for low frequency (target 10-25 trades/year) to avoid fee drag.
# Works in bull markets (breakouts continue) and bear markets (mean reversion to middle band).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    sma_20 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        sma_20[i] = np.mean(close_1w[i-20:i])
    
    std_20 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        std_20[i] = np.std(close_1w[i-20:i])
    
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    middle_band = sma_20  # 20-day SMA
    
    # Align weekly bands to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    middle_aligned = align_htf_to_ltf(prices, df_1w, middle_band)
    
    # Calculate 20-day volume moving average for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need weekly BBands and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: close above upper band with volume confirmation
            if close[i] > upper_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: close below lower band with volume confirmation
            elif close[i] < lower_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: return to middle band
            if close[i] <= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to middle band
            if close[i] >= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBBands20_2.0_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0