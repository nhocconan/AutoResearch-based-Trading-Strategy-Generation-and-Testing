#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsR_Reversion_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === Daily Williams %R (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate highest high and lowest low over 14 days
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close_1d) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), np.nan)
    
    # Align to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 6h Momentum Filter: Price > SMA50 for long bias, < SMA50 for short bias ===
    close_series = pd.Series(prices['close'].values)
    sma50 = close_series.rolling(window=50, min_periods=50).mean().values
    
    # === Volume Confirmation: Volume > 1.5x 20-period average ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        williams_r_val = williams_r_aligned[i]
        sma50_val = sma50[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(williams_r_val) or np.isnan(sma50_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with price above SMA50 and volume confirmation
            if williams_r_val < -80 and close_val > sma50_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with price below SMA50 and volume confirmation
            elif williams_r_val > -20 and close_val < sma50_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 (middle) OR price breaks below SMA50
            if williams_r_val > -50 or close_val < sma50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 (middle) OR price breaks above SMA50
            if williams_r_val < -50 or close_val > sma50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals