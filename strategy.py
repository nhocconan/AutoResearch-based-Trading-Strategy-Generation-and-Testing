#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Williams %R for mean reversion and 1w volume surge for momentum confirmation
# Williams %R identifies overbought/oversold conditions on the daily chart
# Volume surge on weekly chart confirms institutional interest and momentum
# Strategy: Long when daily Williams %R < -80 (oversold) AND weekly volume > 1.5x 20-period average
# Short when daily Williams %R > -20 (overbought) AND weekly volume > 1.5x 20-period average
# Exit when Williams %R returns to neutral range (-50 to -50) or volume normalizes
# Works in both bull and bear markets as it captures mean reversion with momentum confirmation
# Williams %R is effective in ranging markets, volume surge filters for high-probability moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14 periods)
    wr_length = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=wr_length, min_periods=wr_length).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=wr_length, min_periods=wr_length).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)  # Avoid division by zero
    
    # Align Williams %R to 12h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Load 1w data ONCE for volume surge
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w volume surge: current volume > 1.5x 20-period average
    vol_ma_length = 20
    vol_ma = pd.Series(df_1w['volume']).rolling(window=vol_ma_length, min_periods=vol_ma_length).mean().values
    vol_surge = df_1w['volume'].values / vol_ma
    vol_surge = np.where(vol_ma == 0, 0, vol_surge)  # Avoid division by zero
    
    # Align volume surge to 12h timeframe
    vol_surge_aligned = align_htf_to_ltf(prices, df_1w, vol_surge)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need enough for Williams %R and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_aligned[i]) or 
            np.isnan(vol_surge_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr_val = wr_aligned[i]
        vol_surge_val = vol_surge_aligned[i]
        
        if position == 0:
            # Enter long: oversold + volume surge
            if wr_val < -80 and vol_surge_val > 1.5:
                position = 1
                signals[i] = position_size
            # Enter short: overbought + volume surge
            elif wr_val > -20 and vol_surge_val > 1.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral OR volume normalizes
            if wr_val >= -50 or vol_surge_val < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral OR volume normalizes
            if wr_val <= -50 or vol_surge_val < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dWR_1wVolumeSurge_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0