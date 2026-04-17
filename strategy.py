#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Williams %R reversal + volume spike + 12h EMA trend filter.
Long when Williams %R < -80 (oversold) with volume > 2x 20-period average and 12h EMA34 > EMA89.
Short when Williams %R > -20 (overbought) with volume > 2x 20-period average and 12h EMA34 < EMA89.
Exit on opposite Williams %R extreme or trend reversal.
Williams %R identifies exhaustion points; volume spike confirms participation; 12h EMA filter ensures alignment with medium-term trend.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 and EMA89
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = close_12h_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align all to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(89, 20)  # need enough for EMA89 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(ema89_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume and bullish trend (EMA34 > EMA89)
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                ema34_12h_aligned[i] > ema89_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume and bearish trend (EMA34 < EMA89)
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  ema34_12h_aligned[i] < ema89_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R overbought (> -20) or trend turns bearish (EMA34 < EMA89)
            if (williams_r_aligned[i] > -20 or 
                ema34_12h_aligned[i] < ema89_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R oversold (< -80) or trend turns bullish (EMA34 > EMA89)
            if (williams_r_aligned[i] < -80 or 
                ema34_12h_aligned[i] > ema89_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dWilliamsR_VolumeSpike_12hEMA"
timeframe = "6h"
leverage = 1.0