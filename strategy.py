#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d Williams %R extreme filter and volume spike confirmation.
- Primary timeframe: 4h to target 75-200 total trades over 4 years (19-50/year).
- HTF: 1d Williams %R(14) for extreme conditions (oversold < -80 for long, overbought > -20 for short).
- Camarilla levels: H4 and L4 from prior 1d candle (stronger breakout levels than H3/L3).
- Entry: Long when price breaks above prior H4 AND 1d Williams %R < -80 AND volume > 1.8 * volume MA(20).
         Short when price breaks below prior L4 AND 1d Williams %R > -20 AND volume > 1.8 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior 1d close,
        exit short when price crosses above prior 1d close.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy targets strong intraday reversals at key Camarilla levels during extreme market conditions,
designed to work in both bull and bear markets by fading extremes with trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14) for extreme filter
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate prior 1d Camarilla H4 and L4 levels
    # H4 = close + 1.5*(high - low)
    # L4 = close - 1.5*(high - low)
    # Using prior 1d candle to avoid look-ahead
    prior_close = df_1d['close'].shift(1).values
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    camarilla_h4 = prior_close + 1.5 * (prior_high - prior_low)
    camarilla_l4 = prior_close - 1.5 * (prior_high - prior_low)
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)  # for exit condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20)  # Need enough bars for Williams %R and calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(prior_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.8x threshold)
            vol_confirmed = curr_volume > 1.8 * vol_ma[i]
            
            # Long: Price breaks above prior H4 AND Williams %R oversold (< -80) AND volume confirmed
            if curr_close > camarilla_h4_aligned[i] and williams_r_aligned[i] < -80 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior L4 AND Williams %R overbought (> -20) AND volume confirmed
            elif curr_close < camarilla_l4_aligned[i] and williams_r_aligned[i] > -20 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 1d close (mean reversion)
            if curr_close < prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 1d close (mean reversion)
            if curr_close > prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_1dWilliamsR_Extreme_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0