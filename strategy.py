#12h_Camarilla_Pivot_Breakout_Volume
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    R1 = pivot + (range_prev * 1.1 / 12)
    R2 = pivot + (range_prev * 1.1 / 6)
    R3 = pivot + (range_prev * 1.1 / 4)
    S1 = pivot - (range_prev * 1.1 / 12)
    S2 = pivot - (range_prev * 1.1 / 6)
    S3 = pivot - (range_prev * 1.1 / 4)
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Get 12h volume data for confirmation
    volume = prices['volume'].values
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=10).mean().values  # 24 periods = 12 days
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if any pivot value is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = prices['close'].iloc[i]
        vol_ratio = volume[i] / avg_volume[i] if avg_volume[i] > 0 else 0
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close_val > R1_aligned[i] and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close_val < S1_aligned[i] and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below pivot or R2
            if close_val < pivot_aligned[i] or close_val > R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot or S2
            if close_val > pivot_aligned[i] or close_val < S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h Camarilla Pivot Breakout with Volume Confirmation
# Uses daily Camarilla pivot levels (R1, S1) as breakout levels
# Enters long when price breaks above R1 with volume > 1.5x average
# Enters short when price breaks below S1 with volume > 1.5x average
# Exits when price returns to pivot level or reaches R2/S2
# Volume confirmation reduces false breakouts
# Target: 20-40 trades/year on 12h timeframe
name = "12h_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0