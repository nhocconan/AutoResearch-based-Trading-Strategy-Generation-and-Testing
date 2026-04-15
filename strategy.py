#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean-reversion with 1d volume filter
# Williams %R identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought)
# Mean reversion works well in ranging markets; trend filter avoids whipsaws
# Uses 1d average volume to confirm institutional participation
# Designed for low trade frequency (target 20-30/year) with clear entry/exit rules
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams %R(14) on 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # 1d average volume (20-period) for confirmation
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(wr_aligned[i]) or np.isnan(avg_volume_aligned[i]) or 
            np.isnan(volume[i])):
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 1d average volume
        vol_confirmed = volume[i] > 1.5 * avg_volume_aligned[i]
        
        # Mean reversion signals from Williams %R
        oversold = wr_aligned[i] < -80
        overbought = wr_aligned[i] > -20
        
        # Exit when WR returns to neutral zone (-50 to -50)
        neutral_exit = (-50 <= wr_aligned[i] <= -50)
        
        if vol_confirmed:
            # Long when oversold and exiting oversold
            if oversold and wr_aligned[i] > wr_aligned[i-1] and position <= 0:
                position = 1
                signals[i] = position_size
            # Short when overbought and exiting overbought
            elif overbought and wr_aligned[i] < wr_aligned[i-1] and position >= 0:
                position = -1
                signals[i] = -position_size
            # Exit when WR returns to neutral
            elif position == 1 and wr_aligned[i] >= -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and wr_aligned[i] <= -50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_WilliamsR_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0