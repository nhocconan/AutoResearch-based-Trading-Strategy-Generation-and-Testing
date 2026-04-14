#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Fisher Transform (Ehlers) with 1-day volume regime filter
# Long when Fisher crosses above -1.5 in low volatility regime (mean reversion bounce)
# Short when Fisher crosses below +1.5 in low volatility regime (mean reversion fade)
# Exit when Fisher crosses zero (mean reversion complete)
# Fisher Transform identifies extreme price movements likely to revert, volume filter avoids trending markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing reversals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h and 1d data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Fisher Transform on 6h (Ehlers, length=10)
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Normalize price to -1 to +1 range over lookback period
    length = 10
    highest_high = pd.Series(high_6h).rolling(window=length, min_periods=length).max().values
    lowest_low = pd.Series(low_6h).rolling(window=length, min_periods=length).min().values
    
    # Avoid division by zero
    diff = highest_high - lowest_low
    diff = np.where(diff == 0, 1e-10, diff)
    
    # Value from -1 to +1
    value = 2 * ((close_6h - lowest_low) / diff) - 1
    value = np.clip(value, -0.999, 0.999)  # Prevent log domain errors
    
    # Fisher Transform
    fisher = np.zeros_like(value)
    for i in range(1, len(value)):
        fisher[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fisher[i-1]
    
    # Calculate 1-day volume regime (low volume = mean reversion favorable)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_1d  # Current volume vs average
    
    # Align indicators to 6h timeframe
    fisher_aligned = align_htf_to_ltf(prices, df_6h, fisher)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(fisher_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        fisher_val = fisher_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # Only trade in low volume regime (mean reversion environment)
        if vol_ratio < 1.2:  # Below average volume
            if position == 0:
                # Long when Fisher crosses above -1.5 (oversold bounce)
                if fisher_val > -1.5 and fisher_aligned[i-1] <= -1.5:
                    position = 1
                    signals[i] = position_size
                # Short when Fisher crosses below +1.5 (overbought fade)
                elif fisher_val < 1.5 and fisher_aligned[i-1] >= 1.5:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long when Fisher crosses zero (mean reversion complete)
                if fisher_val < 0 and fisher_aligned[i-1] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short when Fisher crosses zero (mean reversion complete)
                if fisher_val > 0 and fisher_aligned[i-1] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            # High volume regime - likely trending, stay flat
            signals[i] = 0.0
            position = 0  # Force flat in trending markets
    
    return signals

name = "6h_Fisher_1dVolumeRegime"
timeframe = "6h"
leverage = 1.0