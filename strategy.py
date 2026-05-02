#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d Williams %R filter and volume confirmation
# Williams %R(14) on 1d identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# Long when price breaks above Camarilla R3 + 1d Williams %R < -80 (oversold) + volume spike
# Short when price breaks below Camarilla S3 + 1d Williams %R > -20 (overbought) + volume spike
# This mean-reversion filter within breakout logic avoids chasing extended moves
# Target: 75-125 total trades over 4 years (19-31/year) - within proven winning range for 6h
# Discrete position sizing: 0.25 to minimize fee churn while maintaining edge

name = "6h_Camarilla_R3_S3_Breakout_1dWilliamsR_OverboughtOversold_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14): (highest_high - close) / (highest_high - lowest_low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling highest high and lowest low for 14-period Williams %R
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    williams_r = np.where(hl_range != 0, ((highest_high - close_1d) / hl_range) * -100, -50.0)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R3 = close + 1.500*(high-low), S3 = close - 1.500*(high-low)
    camarilla_high = close_1d + 1.500 * (high_1d - low_1d)  # R3 level
    camarilla_low = close_1d - 1.500 * (high_1d - low_1d)   # S3 level
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Calculate volume spike (2.0x 20-period average) - balanced threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, volume MA and HTF data alignment)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + 1d Williams %R < -80 (oversold) + volume spike
            if close[i] > camarilla_high_aligned[i] and williams_r_aligned[i] < -80 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 + 1d Williams %R > -20 (overbought) + volume spike
            elif close[i] < camarilla_low_aligned[i] and williams_r_aligned[i] > -20 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 (reversal signal)
            if close[i] < camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 (reversal signal)
            if close[i] > camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals