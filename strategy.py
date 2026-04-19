#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R for mean-reversion signals in trending markets.
# Williams %R > -20 indicates overbought (short signal), < -80 indicates oversold (long signal).
# Filtered by 4-hour EMA50 trend to avoid counter-trend trades.
# Volume confirmation (volume > 1.5x 20-period average) ensures participation.
# Designed for both bull and bear markets by aligning with higher-timeframe trend.
# Target: 20-40 trades/year per symbol with disciplined entries.
name = "4h_WilliamsR_1d_EMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 4-hour EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), above EMA50, with volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), below EMA50, with volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Williams %R rises above -50 or price breaks below EMA50
            if (williams_r_aligned[i] > -50) or (close[i] < ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Williams %R falls below -50 or price breaks above EMA50
            if (williams_r_aligned[i] < -50) or (close[i] > ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals