#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining daily Camarilla pivot levels with 6h price action and volume confirmation.
# Camarilla levels (R3/S3, R4/S4) from prior day provide institutional support/resistance.
# Long when price closes above R3 with volume > 1.5x average, targeting R4.
# Short when price closes below S3 with volume > 1.5x average, targeting S4.
# Exit when price reaches target level or reverses back through R3/S3.
# Uses discrete position sizing (0.25) to limit drawdown and minimize fee churn.
# Designed to work in both trending and ranging markets by fading extremes and capturing breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day using prior day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.5 * range_1d
    camarilla_r3 = close_1d + 1.1 * range_1d
    camarilla_s3 = close_1d - 1.1 * range_1d
    camarilla_s4 = close_1d - 1.5 * range_1d
    
    # Shift to use prior day's levels (avoid look-ahead)
    camarilla_r4 = np.roll(camarilla_r4, 1)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_s4 = np.roll(camarilla_s4, 1)
    camarilla_r4[0] = np.nan
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    camarilla_s4[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 1.5x average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after enough data for volume MA
        # Skip if any critical data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for new positions
            # Long: close above R3 with volume confirmation
            if close[i] > r3_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Short: close below S3 with volume confirmation
            elif close[i] < s3_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Manage long position
            # Exit if: reaches R4 target OR reverses back below R3
            if close[i] >= r4_aligned[i] or close[i] <= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Manage short position
            # Exit if: reaches S4 target OR reverses back above S3
            if close[i] <= s4_aligned[i] or close[i] >= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_Volume"
timeframe = "6h"
leverage = 1.0