# #!/usr/bin/env python3
# Strategy: 12h_1w_Camarilla_Pivot_R1S1_Breakout_Volume
# Hypothesis: Weekly Camarilla pivot levels (R1/S1) act as strong support/resistance.
# Long when price breaks above weekly R1 with volume > 1.5x daily average volume.
# Short when price breaks below weekly S1 with volume > 1.5x daily average volume.
# Exit when price crosses back through the Camarilla pivot point (CP).
# Uses weekly timeframe for structure to avoid whipsaw, daily volume for confirmation.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).
# Works in bull markets (breakouts continue) and bear markets (reversions at weak levels).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous week
    # Formula: CP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # We use the previous week's values (shifted by 1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    cp = (high_1w + low_1w + close_1w) / 3
    r1 = close_1w + (high_1w - low_1w) * 1.1 / 12
    s1 = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align to 12h timeframe (these levels are valid for the entire week after it closes)
    cp_aligned = align_htf_to_ltf(prices, df_1w, cp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily average volume for confirmation (20-day average)
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Need volume MA and at least one week of data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        cp_val = cp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Volume confirmation: current volume > 1.5x daily average
        volume_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: break above weekly R1 with volume confirmation
            if price > r1_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly S1 with volume confirmation
            elif price < s1_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below Camarilla pivot point
            if price < cp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above Camarilla pivot point
            if price > cp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals