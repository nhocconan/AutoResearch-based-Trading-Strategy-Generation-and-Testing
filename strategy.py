#!/usr/bin/env python3
# 12h_1d_Alligator_Trend_With_Volume
# Hypothesis: Use Williams Alligator on 1d to determine trend (Green line > Red = uptrend, Red > Green = downtrend).
# On 12h, trade in direction of 1d trend: long when price > Jaw (blue) in uptrend, short when price < Jaw in downtrend.
# Requires volume confirmation (volume > 1.5x 20-period average) to avoid false breakouts.
# Exit when price crosses the Jaw in opposite direction or volume drops.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear via 1d trend filter - only trade with the higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Alligator_Trend_With_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # === Calculate Williams Alligator on 1d ===
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    close_1d = df_1d['close'].values
    
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_1d = smma(close_1d, 13)  # Blue line
    teeth_1d = smma(close_1d, 8)  # Red line
    lips_1d = smma(close_1d, 5)   # Green line
    
    # Shift as per Alligator specification
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Trend determination: Green > Red = uptrend, Red > Green = downtrend
    # We'll use lips > teeth for uptrend, teeth > lips for downtrend
    trend_up_1d = lips_1d > teeth_1d
    trend_down_1d = teeth_1d > lips_1d
    
    # Align all 1d indicators to 12h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d.astype(float))
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d.astype(float))
    
    # === 12h: Jaw (blue line) for entry/exit ===
    close_12h = prices['close'].values
    jaw_12h = smma(close_12h, 13)
    jaw_12h = np.roll(jaw_12h, 8)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        jaw_12h_val = jaw_12h[i]
        jaw_1d_val = jaw_1d_aligned[i]
        teeth_1d_val = teeth_1d_aligned[i]
        lips_1d_val = lips_1d_aligned[i]
        trend_up_1d_val = trend_up_1d_aligned[i]
        trend_down_1d_val = trend_down_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_12h_val) or np.isnan(jaw_1d_val) or np.isnan(teeth_1d_val) or 
            np.isnan(lips_1d_val) or np.isnan(trend_up_1d_val) or np.isnan(trend_down_1d_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d uptrend (Green > Red) AND price > 12h Jaw with volume confirmation
            if (trend_up_1d_val and  # 1d uptrend: Lips > Teeth
                close_val > jaw_12h_val and  # Price above 12h Jaw
                vol_ratio_val > 1.5):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend (Red > Green) AND price < 12h Jaw with volume confirmation
            elif (trend_down_1d_val and  # 1d downtrend: Teeth > Lips
                  close_val < jaw_12h_val and  # Price below 12h Jaw
                  vol_ratio_val > 1.5):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: 1d trend turns down OR price crosses below 12h Jaw
            if (not trend_up_1d_val or  # 1d trend turned down
                close_val < jaw_12h_val):  # Price crossed below 12h Jaw
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: 1d trend turns up OR price crosses above 12h Jaw
            if (not trend_down_1d_val or  # 1d trend turned up
                close_val > jaw_12h_val):  # Price crossed above 12h Jaw
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals