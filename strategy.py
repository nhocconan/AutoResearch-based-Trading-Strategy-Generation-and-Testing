#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d Bollinger Band mean reversion + volume confirmation
# Williams %R identifies overbought/oversold conditions on 6h timeframe
# 1d Bollinger Bands provide higher timeframe mean reversion context (price near bands)
# Volume confirmation ensures sufficient participation at reversal points
# Designed for mean reversion in ranging markets with trend filters to avoid whipsaws
name = "6h_WilliamsR_1dBB_MeanRev_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Bollinger Bands for mean reversion context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Williams %R calculation on 6h (14 period)
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, -100 * ((highest_high - close) / hh_ll), -50)
    
    # Volume confirmation: volume > 1.2 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(williams_period, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(sma_20_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price near lower BB + volume confirmation
            if (williams_r[i] < -80 and 
                close[i] <= lower_bb_aligned[i] * 1.02 and  # Allow small tolerance
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price near upper BB + volume confirmation
            elif (williams_r[i] > -20 and 
                  close[i] >= upper_bb_aligned[i] * 0.98 and  # Allow small tolerance
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Williams %R rises above -50 (momentum fading) or price reaches middle band
            if (williams_r[i] > -50) or (close[i] >= sma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Williams %R falls below -50 (momentum fading) or price reaches middle band
            if (williams_r[i] < -50) or (close[i] <= sma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals