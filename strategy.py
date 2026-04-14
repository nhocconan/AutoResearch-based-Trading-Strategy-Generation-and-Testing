#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day 200-day Simple Moving Average (SMA) as long-term trend filter
# and 1-day Bollinger Bands for mean reversion entries. Long when price crosses below lower BB
# in an uptrend (price > SMA200), short when price crosses above upper BB in a downtrend
# (price < SMA200). Exit when price returns to SMA200 or Bollinger Band middle.
# Uses volatility-adjusted bands (2 standard deviations) to adapt to market conditions.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for SMA200 and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need enough for SMA(200)
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 200-day Simple Moving Average
    sma_200 = np.full_like(close_1d, np.nan)
    for i in range(199, len(close_1d)):
        sma_200[i] = np.mean(close_1d[i-199:i+1])
    
    # Calculate Bollinger Bands (20-period, 2 standard deviations)
    sma_20 = np.full_like(close_1d, np.nan)
    std_20 = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-19:i+1])
        std_20[i] = np.std(close_1d[i-19:i+1])
    
    # Upper and Lower Bollinger Bands
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20  # 20-period SMA
    
    # Align indicators to 6h timeframe
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need 200 days for SMA200)
    start = 200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma_200_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(middle_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 200-day SMA
        above_sma200 = close[i] > sma_200_aligned[i]
        below_sma200 = close[i] < sma_200_aligned[i]
        
        if position == 0:
            # Look for mean reversion entries
            # Long: price crosses below lower BB AND in uptrend (price > SMA200)
            if (close[i] < lower_bb_aligned[i] and 
                above_sma200):
                position = 1
                signals[i] = position_size
            # Short: price crosses above upper BB AND in downtrend (price < SMA200)
            elif (close[i] > upper_bb_aligned[i] and 
                  below_sma200):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or crosses above SMA200 (trend change)
            if (close[i] >= middle_bb_aligned[i] or 
                close[i] >= sma_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle BB or crosses below SMA200 (trend change)
            if (close[i] <= middle_bb_aligned[i] or 
                close[i] <= sma_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_SMA200_BollingerBands_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0