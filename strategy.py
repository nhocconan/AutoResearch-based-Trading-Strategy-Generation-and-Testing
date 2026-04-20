#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_DonchianBreakout_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:  # Need enough for Donchian
        return np.zeros(n)
    
    # === Weekly Donchian channel (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (only available after weekly bar closes)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, high_roll)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, low_roll)
    
    # === Daily volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        current_close = prices['close'].iloc[i]
        current_volume = volume[i]
        current_vol_ma = vol_ma[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper) or np.isnan(lower) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-day average
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        if position == 0:
            # Long: break above weekly Donchian upper with volume
            if current_close > upper and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: break below weekly Donchian lower with volume
            elif current_close < lower and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: close below weekly Donchian lower
            if current_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly Donchian upper
            if current_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals