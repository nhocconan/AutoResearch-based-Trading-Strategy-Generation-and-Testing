#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Pivot Point (Classic) R1/S1 breakout with weekly volume confirmation.
# Long when price breaks above R1 pivot AND weekly volume > 1.5x weekly average volume.
# Short when price breaks below S1 pivot AND weekly volume > 1.5x weekly average volume.
# Exit when price returns to the daily pivot point (PP).
# Uses daily pivot for structure, weekly volume for confirmation to reduce false signals.
# Target: 10-20 trades/year per symbol.
name = "1d_DailyPivot_R1S1_Breakout_WeeklyVolume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points (classic formula)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    
    # Get weekly data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly average volume (20-period)
    vol_ma_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for weekly volume MA
    
    for i in range(start_idx, n):
        # Skip if weekly volume data not available
        if np.isnan(vol_ma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1w_aligned[i]
        pivot = pp[i]
        r1_level = r1[i]
        s1_level = s1[i]
        
        # Volume confirmation: weekly volume > 1.5x weekly average
        vol_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation
            if price > r1_level and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation
            elif price < s1_level and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to or below pivot point
            if price <= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to or above pivot point
            if price >= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals