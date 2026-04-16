#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 6h volume confirmation.
# Long when Williams %R < -80 (oversold) AND volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. Williams %R identifies reversal points in ranging markets.
# Volume confirmation ensures breakouts have participation. 6h timeframe targets 50-150 total trades over 4 years.
# Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies).

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to primary timeframe (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 6h Indicators: Volume average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        vol = volume[i]
        vol_avg = vol_ma[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Williams %R crosses above -50 (momentum fading)
            if wr > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses below -50 (momentum fading)
            if wr < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Oversold (WR < -80) AND volume confirmation (vol > 1.5x average)
            if (wr < -80) and (vol > 1.5 * vol_avg):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Overbought (WR > -20) AND volume confirmation (vol > 1.5x average)
            elif (wr > -20) and (vol > 1.5 * vol_avg):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dWilliamsR_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0