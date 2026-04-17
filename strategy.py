#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1-week Bollinger Band mean reversion in trending markets.
- In uptrends (price > 200-day MA): buy dips to lower Bollinger Band (20, 2) with volume confirmation.
- In downtrends (price < 200-day MA): sell rallies to upper Bollinger Band with volume confirmation.
- Uses 1-week Bollinger Bands to reduce noise and 1-day 200-period MA for trend filter.
- Volume filter: current volume > 1.5x 20-period average on 1-day timeframe.
- Position size: 0.25 for entries, 0 for exits.
Target: 20-50 total trades over 4 years (5-12/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (200-day MA) and volume filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 200-day moving average for trend filter
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Volume filter: 1.5x 20-period average on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Bollinger Bands (20, 2)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Bollinger Bands (20, 2) on weekly
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + 2 * std_20_1w
    lower_bb_1w = sma_20_1w - 2 * std_20_1w
    
    # Align all to 12h timeframe
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need enough data for 200-day MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_200_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: price above/below 200-day MA
        uptrend = close[i] > sma_200_aligned[i]
        downtrend = close[i] < sma_200_aligned[i]
        
        if position == 0:
            # Long: price at lower BB, volume spike, uptrend
            if (close[i] <= lower_bb_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price at upper BB, volume spike, downtrend
            elif (close[i] >= upper_bb_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                  downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above the 20-day SMA (middle of BB)
            if close[i] >= sma_20_1w_aligned[i] if 'sma_20_1w_aligned' in locals() else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below the 20-day SMA (middle of BB)
            if close[i] <= sma_20_1w_aligned[i] if 'sma_20_1w_aligned' in locals() else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    # Pre-calculate the 20-week SMA for exit condition to avoid recomputation in loop
    # This is done outside the loop for efficiency
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Re-run the loop with pre-calculated SMA for exit
    signals = np.zeros(n)
    position = 0
    
    for i in range(start_idx, n):
        if (np.isnan(sma_200_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        uptrend = close[i] > sma_200_aligned[i]
        downtrend = close[i] < sma_200_aligned[i]
        
        if position == 0:
            if (close[i] <= lower_bb_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                uptrend):
                signals[i] = 0.25
                position = 1
            elif (close[i] >= upper_bb_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                  downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            if close[i] >= sma_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] <= sma_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wBBands20_2_1dMA200_Volume"
timeframe = "12h"
leverage = 1.0