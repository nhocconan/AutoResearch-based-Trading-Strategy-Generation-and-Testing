#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1w Williams Alligator (Jaw/Teeth/Lips) for trend direction and 1d volume spike for confirmation.
- Uses 1w for trend filter (Alligator aligned = trending market) and 1d for volume confirmation (>2.0x average)
- 4h only for entry timing to reduce whipsaw and improve trade frequency
- Session filter: 08-20 UTC to avoid low-liquidity periods
- Position size: 0.30 (discrete level to minimize fee churn)
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter (Alligator alignment) and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 2.0x 24-period average (strict for 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1w Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs of median price, shifted)
    df_1w = get_htf_data(prices, '1w')
    median_1w = (df_1w['high'].values + df_1w['low'].values) / 2.0
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_1w = pd.Series(median_1w).rolling(window=13, min_periods=13).mean().values
    jaw_1w = np.roll(jaw_1w, 8)  # shift 8 bars forward
    jaw_1w[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_1w = pd.Series(median_1w).rolling(window=8, min_periods=8).mean().values
    teeth_1w = np.roll(teeth_1w, 5)  # shift 5 bars forward
    teeth_1w[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips_1w = pd.Series(median_1w).rolling(window=5, min_periods=5).mean().values
    lips_1w = np.roll(lips_1w, 3)  # shift 3 bars forward
    lips_1w[:3] = np.nan
    
    # Align Alligator lines to 4h timeframe (use prior completed 1w bar)
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 13)  # volume MA, Alligator Jaw
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw_1w_aligned[i]) or
            np.isnan(teeth_1w_aligned[i]) or
            np.isnan(lips_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Alligator alignment: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        jaw = jaw_1w_aligned[i]
        teeth = teeth_1w_aligned[i]
        lips = lips_1w_aligned[i]
        uptrend = jaw > teeth and teeth > lips
        downtrend = jaw < teeth and teeth < lips
        
        if position == 0:
            # Long: Alligator uptrend AND volume confirmation AND in session
            if uptrend and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Alligator downtrend AND volume confirmation AND in session
            elif downtrend and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Alligator trend breaks down (Jaw <= Teeth OR Teeth <= Lips)
            if jaw <= teeth or teeth <= lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Alligator trend breaks up (Jaw >= Teeth OR Teeth >= Lips)
            if jaw >= teeth or teeth >= lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_WilliamsAlligator_Trend_1dVolumeSpike_Session"
timeframe = "4h"
leverage = 1.0