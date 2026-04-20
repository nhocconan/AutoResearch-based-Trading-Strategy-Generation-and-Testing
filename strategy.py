#!/usr/bin/env python3
"""
12h_1w_Alligator_ElderRay_Trend_v1
Concept: 12h price follows 1-week Alligator + Elder Ray trend with volume confirmation.
- Long: 1w Alligator bullish (jaw < teeth < lips) AND Elder Ray bullish (Bull Power > 0) AND 1d volume > 1.5x 20-period avg
- Short: 1w Alligator bearish (jaw > teeth > lips) AND Elder Ray bearish (Bear Power < 0) AND 1d volume > 1.5x 20-period avg
- Exit: Alligator trend reversal (jaw crosses teeth) OR Elder Ray momentum divergence
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear: Alligator defines trend, Elder Ray measures momentum strength, volume confirms conviction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Alligator_ElderRay_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly: Alligator (13,8,5) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    jaw_values = jaw.values
    
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    teeth_values = teeth.values
    
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    lips_values = lips.values
    
    # Align Alligator lines
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_values)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_values)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_values)
    
    # === Weekly: Elder Ray (13-period EMA) ===
    ema13 = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1w - ema13  # Bull Power = High - EMA13
    bear_power = low_1w - ema13   # Bear Power = Low - EMA13
    
    # Align Elder Ray
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # === Daily: Volume MA (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 12h: Price ===
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma_20 = vol_ma_20_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(bull_power_val) or np.isnan(bear_power_val) or np.isnan(vol_ma_20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-period average
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 1.5 * vol_ma_20
        
        if position == 0:
            # Alligator bullish: jaw < teeth < lips
            alligator_bullish = jaw_val < teeth_val < lips_val
            # Alligator bearish: jaw > teeth > lips
            alligator_bearish = jaw_val > teeth_val > lips_val
            
            # Long: Alligator bullish AND Elder Ray bullish (Bull Power > 0) AND volume confirmation
            if alligator_bullish and bull_power_val > 0 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Elder Ray bearish (Bear Power < 0) AND volume confirmation
            elif alligator_bearish and bear_power_val < 0 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator trend reversal (jaw crosses above teeth) OR Elder Ray weakness (Bull Power <= 0)
            if jaw_val >= teeth_val or bull_power_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator trend reversal (jaw crosses below teeth) OR Elder Ray weakness (Bear Power >= 0)
            if jaw_val <= teeth_val or bear_power_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals