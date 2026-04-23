#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator + Elder Ray combo with 1w EMA50 trend filter and volume spike confirmation.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
  Long when Lips > Teeth > Jaw (bullish alignment)
  Short when Lips < Teeth < Jaw (bearish alignment)
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
  Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
- 1w EMA50: Strong HTF trend filter to avoid counter-trend trades
- Volume confirmation: > 2.0x 20-period average for conviction
- Exit: Opposite Alligator alignment or EMA50 trend flip
- Uses Alligator for trend structure, Elder Ray for momentum, 1w EMA50 for HTF filter
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (Alligator alignment up) and bear (Alligator alignment down) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: SMAs on median price (typical price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(8, min_periods=8).mean().values  # SMA(13) then SMA(8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(5, min_periods=5).mean().values   # SMA(8) then SMA(5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(3, min_periods=3).mean().values   # SMA(5) then SMA(3)
    
    # Elder Ray: EMA13 for power calculations
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Smooth Elder Ray powers for confirmation (3-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need 50 for EMA50, 20 for volume MA, 13 for EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Elder Ray confirmation
        bull_power_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_power_falling = bear_power_smooth[i] < bear_power_smooth[i-1]
        
        if position == 0:
            # Long: Bullish Alligator + Elder Ray bullish + volume confirmation + price > 1w EMA50
            if (bullish_alignment and 
                bull_power_smooth[i] > 0 and 
                bull_power_rising and 
                volume_confirm and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + Elder Ray bearish + volume confirmation + price < 1w EMA50
            elif (bearish_alignment and 
                  bear_power_smooth[i] < 0 and 
                  bear_power_falling and 
                  volume_confirm and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR price < 1w EMA50 (trend flip)
            if bearish_alignment or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR price > 1w EMA50 (trend flip)
            if bullish_alignment or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_ElderRay_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0