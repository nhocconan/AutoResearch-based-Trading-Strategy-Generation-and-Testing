#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R + 1d Volume Regime Filter
    # Long: Williams %R < -80 (oversold) AND 1d volume > 1.5 * 20-period MA (volume expansion)
    # Short: Williams %R > -20 (overbought) AND 1d volume > 1.5 * 20-period MA
    # Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
    # Uses 6h for Williams %R (momentum extremes), 1d for volume regime (institutional participation)
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for Williams %R (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h Williams %R (%R = (Highest High - Close) / (Highest High - Lowest Low) * -100)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # 14-period Williams %R
    period = 14
    highest_high = pd.Series(high_6h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_6h).rolling(window=period, min_periods=period).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close_6h) / (highest_high - lowest_low), 
                          -50)  # neutral when range=0
    
    # Align 6h Williams %R to 6h timeframe (no additional delay for price-based indicators)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate 1d Volume Regime: volume > 1.5 * 20-period MA
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1d > (1.5 * vol_ma_20)
    
    # Align 1d volume regime to 6h (wait for completed 1d bar)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime filter: only trade when 1d volume is elevated
        high_volume = volume_regime_aligned[i] > 0.5
        
        # Williams %R signals
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        exit_long = williams_r_aligned[i] > -50  # exit long when crosses above -50
        exit_short = williams_r_aligned[i] < -50  # exit short when crosses below -50
        
        # Entry logic: Williams %R extreme + high volume regime
        long_entry = oversold and high_volume
        short_entry = overbought and high_volume
        
        # Exit logic: Williams %R crosses midpoint
        long_exit = exit_long
        short_exit = exit_short
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williams_r_volume_regime_v1"
timeframe = "6h"
leverage = 1.0