#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate Camarilla pivot levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # Resistance levels: R1 = close + (range * 1.1/12), R2 = close + (range * 1.1/6), etc.
    # Support levels: S1 = close - (range * 1.1/12), S2 = close - (range * 1.1/6), etc.
    daily_range = high_1d - low_1d
    
    # Key levels for breakout: R3, R4, S3, S4 (most significant)
    r3 = close_1d + (daily_range * 1.1 / 4)
    r4 = close_1d + (daily_range * 1.1 / 2)
    s3 = close_1d - (daily_range * 1.1 / 4)
    s4 = close_1d - (daily_range * 1.1 / 2)
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > r4_aligned[i]  # Break above R4
        breakout_down = price_close < s4_aligned[i]  # Break below S4
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above R4 with volume confirmation
        if breakout_up and vol_confirm:
            enter_long = True
        
        # Short: Break below S4 with volume confirmation
        if breakout_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level touch
        exit_long = price_close < s3_aligned[i]  # Return to S3 level
        exit_short = price_close > r3_aligned[i]  # Return to R3 level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h Camarilla breakout strategy using daily pivot levels.
# Enters long when price breaks above R4 with volume confirmation, short when breaks below S4.
# Exits when price returns to S3/R3 levels respectively.
# Uses volume confirmation to avoid false breakouts and Camarilla levels for precise entry/exit.
# Works in both bull and bear markets by capturing significant breakouts in either direction.
# Position size 0.25 limits drawdown during volatile periods.
# Target: 20-40 trades per year (80-160 total over 4 years) to minimize fee drag.