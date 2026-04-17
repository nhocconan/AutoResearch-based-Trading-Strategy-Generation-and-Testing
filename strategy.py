#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Williams %R mean reversion + volume spike + 6h EMA34 trend filter.
Long when Williams %R < -80 (oversold), volume > 2x 20-period average, and price > EMA34.
Short when Williams %R > -20 (overbought), volume > 2x 20-period average, and price < EMA34.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) or volume drops.
Designed to capture mean reversion in extreme conditions with institutional volume confirmation,
using 1d Williams %R for higher-timeframe extreme readings and 6h for execution.
Williams %R identifies overextended moves that tend to reverse, especially when combined with volume spikes.
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
    
    # Get 1d data for Williams %R calculation (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    
    # Calculate 6h EMA34 for trend filter
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Williams %R to 6h timeframe (with extra delay for indicator confirmation)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R calculation and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema34[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), volume spike, and uptrend (price > EMA34)
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                close[i] > ema34[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), volume spike, and downtrend (price < EMA34)
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  close[i] < ema34[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (recovering from oversold) OR volume drops
            if (williams_r_aligned[i] > -50 or 
                volume[i] <= vol_ma_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (recovering from overbought) OR volume drops
            if (williams_r_aligned[i] < -50 or 
                volume[i] <= vol_ma_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dWilliamsR_MeanReversion_Volume_EMA34_Trend"
timeframe = "6h"
leverage = 1.0