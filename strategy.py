#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with Volume Confirmation
# Long when: price > Alligator Jaw, Bull Power > 0, Bear Power < 0, volume > 1.5x avg
# Short when: price < Alligator Jaw, Bull Power < 0, Bear Power > 0, volume > 1.5x avg
# Exit when: price crosses Alligator Teeth or power signals reverse
# Williams Alligator identifies trend, Elder Ray measures bull/bear power, volume confirms conviction.
# Designed for 12h timeframe to capture multi-day moves with low frequency (target: 50-150 trades over 4 years).

name = "12h_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # === Williams Alligator (13,8,5 SMAs shifted) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Jaw (13-period SMA, shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMA, shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === Elder Ray (13-period EMA) ===
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    ema13 = pd.Series(daily_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    bull_power = daily_high - ema13
    bear_power = daily_low - ema13
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Get values
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price > Jaw, Bull Power > 0, Bear Power < 0, volume confirmation
            if close_val > jaw_val and bull_val > 0 and bear_val < 0 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Jaw, Bull Power < 0, Bear Power > 0, volume confirmation
            elif close_val < jaw_val and bull_val < 0 and bear_val > 0 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Teeth or power signals reverse
            if close_val < teeth_val or bull_val <= 0 or bear_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Teeth or power signals reverse
            if close_val > teeth_val or bull_val >= 0 or bear_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals