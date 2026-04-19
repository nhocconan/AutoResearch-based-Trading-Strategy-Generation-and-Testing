#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 1D Camarilla Pivot Range (R1/S1) with Volume Confirmation
# - Uses 1D Camarilla pivot levels (R1, S1) calculated from previous day's range
# - Long when price breaks above R1 with volume confirmation
# - Short when price breaks below S1 with volume confirmation
# - Exits when price returns to the pivot range (between S1 and R1)
# - Camarilla levels act as natural support/resistance with high probability of reversal/continuation
# - Volume confirmation ensures breakouts have institutional participation
# - Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency
name = "6h_1D_Camarilla_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # Where C, H, L are from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use only previous day's data
    cam_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    cam_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 6h timeframe (wait for 1D bar to close)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cam_r1_aligned[i]) or np.isnan(cam_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > cam_r1_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < cam_s1_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit when price returns to or below R1 (mean reversion to pivot level)
            if close[i] <= cam_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit when price returns to or above S1 (mean reversion to pivot level)
            if close[i] >= cam_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals