#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend(ATR=10, mult=3.0) + 1d Camarilla Pivot Breakout + Volume Spike
# Supertrend identifies the primary trend direction on 6h to avoid counter-trend trades
# Camarilla levels from 1d: Break above R3 or below S3 with volume spike triggers entry in trend direction
# Volume confirmation (2.0x 20-period avg) ensures momentum behind breakout
# Discrete position sizing (0.25) minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe
# Works in bull markets via buying breakouts in uptrend and in bear markets via selling breakdowns in downtrend

name = "6h_Supertrend_1dCamarilla_R3S3_Breakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using typical formula: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Simplified: R3 = close + 1.21*(high-low)/4, S3 = close - 1.21*(high-low)/4
    # Further simplified: R3 = close + 0.3025*(high-low), S3 = close - 0.3025*(high-low)
    # Commonly used: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using standard Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * camarilla_range / 2
    s3_1d = close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate Supertrend on 6h data (ATR=10, mult=3.0)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(10)
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Start from index 10 (after ATR warmup)
    for i in range(10, n):
        if np.isnan(atr[i]) or np.isnan(hl2[i]):
            continue
            
        # Supertrend logic
        if i == 10:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            if close[i-1] > supertrend[i-1]:
                # Previous close was above previous Supertrend → uptrend
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                # Previous close was below previous Supertrend → downtrend
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for ATR and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(supertrend[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 + volume spike + 6h uptrend (Supertrend up)
            if (close[i] > r3_1d_aligned[i] and volume_spike[i] and direction[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + volume spike + 6h downtrend (Supertrend down)
            elif (close[i] < s3_1d_aligned[i] and volume_spike[i] and direction[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Supertrend (trend reversal) or below S3
            if close[i] < supertrend[i] or close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Supertrend (trend reversal) or above R3
            if close[i] > supertrend[i] or close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals