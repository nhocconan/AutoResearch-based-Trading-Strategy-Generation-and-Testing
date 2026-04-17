#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R oscillator for mean reversion in ranging markets.
Long when 1d Williams %R crosses above -80 (oversold) with volume confirmation and price > 4h EMA200 (trend filter).
Short when 1d Williams %R crosses below -20 (overbought) with volume confirmation and price < 4h EMA200.
Uses 1d Williams %R for institutional-grade overbought/oversold levels, EMA200 for trend alignment,
and volume to confirm reversal strength. Designed to work in ranging markets (chop) and avoid strong trends.
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
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate 4h EMA200 for trend filter
    close_s = pd.Series(close)
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema200[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Williams %R signals: cross above -80 (oversold) or below -20 (overbought)
        wr_prev = williams_r_aligned[i-1] if i > 0 else -50
        wr_curr = williams_r_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below (exiting oversold) with volume and mild uptrend (price > EMA200)
            if (wr_prev <= -80 and wr_curr > -80 and 
                volume_confirmed and 
                close[i] > ema200[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (exiting overbought) with volume and mild downtrend (price < EMA200)
            elif (wr_prev >= -20 and wr_curr < -20 and 
                  volume_confirmed and 
                  close[i] < ema200[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) or price breaks above EMA200 too strongly
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) or price breaks below EMA200 too strongly
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_MeanReversion_Volume_EMA200_TrendFilter"
timeframe = "4h"
leverage = 1.0