#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Fibonacci pivot levels from weekly data with volume confirmation
# Fade at 0.618 retracement levels, breakout continuation at 1.272 extension levels
# Uses weekly structure to identify key levels that work across market regimes
# Volume confirms institutional participation at these levels
# Targets 15-25 trades/year to minimize fee drag while capturing significant moves

name = "6h_fib_pivot_weekly_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Fibonacci pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Fibonacci levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly range
    range_1w = high_1w - low_1w
    
    # Fibonacci retracement levels (from weekly high to low)
    fib_0618 = high_1w - 0.618 * range_1w  # 61.8% retracement from high
    fib_382 = high_1w - 0.382 * range_1w   # 38.2% retracement from high
    
    # Fibonacci extension levels (beyond weekly range)
    fib_ext_up = high_1w + 0.272 * range_1w  # 1.272 extension above high
    fib_ext_down = low_1w - 0.272 * range_1w  # 1.272 extension below low
    
    # Align Fibonacci levels to 6h timeframe
    fib_0618_aligned = align_htf_to_ltf(prices, df_1w, fib_0618)
    fib_382_aligned = align_htf_to_ltf(prices, df_1w, fib_382)
    fib_ext_up_aligned = align_htf_to_ltf(prices, df_1w, fib_ext_up)
    fib_ext_down_aligned = align_htf_to_ltf(prices, df_1w, fib_ext_down)
    
    # Volume confirmation (20-period average on 6h = 5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(fib_0618_aligned[i]) or np.isnan(fib_382_aligned[i]) or 
            np.isnan(fib_ext_up_aligned[i]) or np.isnan(fib_ext_down_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Price levels
        fib_0618_level = fib_0618_aligned[i]
        fib_382_level = fib_382_aligned[i]
        fib_ext_up_level = fib_ext_up_aligned[i]
        fib_ext_down_level = fib_ext_down_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price breaks below 0.618 (trend failure) or reaches extension (take profit)
            if close[i] < fib_0618_level or close[i] > fib_ext_up_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price breaks above 0.618 (trend failure) or reaches extension (take profit)
            if close[i] > fib_0618_level or close[i] < fib_ext_down_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Fade at 0.382 level: buy near support, sell near resistance
            # Breakout continuation: break above weekly high or below weekly low with volume
            
            # Long: buy at 0.382 support bounce or break above weekly high extension
            if (abs(close[i] - fib_382_level) < 0.001 * fib_382_level and vol_confirm) or \
               (close[i] > fib_ext_up_level and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: sell at 0.382 resistance bounce or break below weekly low extension
            elif (abs(close[i] - fib_382_level) < 0.001 * fib_382_level and vol_confirm) or \
                 (close[i] < fib_ext_down_level and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals