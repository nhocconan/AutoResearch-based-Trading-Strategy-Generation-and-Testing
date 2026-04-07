#!/usr/bin/env python3
"""
6h_camarilla_pivot_12h_volume_v1
Hypothesis: On 6h timeframe, use Camarilla pivot levels from 12h timeframe for reversal/fade signals with volume confirmation.
Fade at R3/S3 levels (mean reversion), breakout continuation at R4/S4 levels (trend following).
Volume > 1.3x average confirms the move. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_12h_volume_v1"
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
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for Camarilla pivot calculation (calculate once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla formulas: 
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    prev_high = df_12h['high'].shift(1).values  # Previous 12h high
    prev_low = df_12h['low'].shift(1).values    # Previous 12h low
    prev_close = df_12h['close'].shift(1).values # Previous 12h close
    
    # Calculate pivot levels
    r4 = prev_close + 1.5 * (prev_high - prev_low)
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    s4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align to 6h timeframe (shifted by 1 to avoid look-ahead)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after volume MA warmup
        # Skip if required data not available
        if (np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on breakdown below S3 (mean reversion failed)
            if close[i] < s3_aligned[i]:
                exit_long = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on breakout above R3 (mean reversion failed)
            if close[i] > r3_aligned[i]:
                exit_short = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3 (mean reversion)
            fade_long = (close[i] <= s3_aligned[i]) and vol_confirm
            fade_short = (close[i] >= r3_aligned[i]) and vol_confirm
            
            # Breakout continuation at R4/S4 (trend following)
            breakout_long = (close[i] >= r4_aligned[i]) and vol_confirm
            breakout_short = (close[i] <= s4_aligned[i]) and vol_confirm
            
            if fade_long or breakout_long:
                position = 1
                signals[i] = 0.25
            elif fade_short or breakout_short:
                position = -1
                signals[i] = -0.25
    
    return signals