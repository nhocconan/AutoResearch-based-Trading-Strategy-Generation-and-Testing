#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray combo with volume confirmation.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
  Trend up: Lips > Teeth > Jaw; Trend down: Lips < Teeth < Jaw
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
  Bullish: Bull Power > 0 and rising; Bearish: Bear Power < 0 and falling
- Entry: Alligator aligned + Elder Ray confirmation + volume > 1.5x 20-period avg
- Exit: Alligator reversal OR Elder Ray divergence
- Uses 1d HTF for trend filter (price > 1d EMA50 for longs, < for shorts)
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (catch trends) and bear (fade counter-trend moves)
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
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Williams Alligator SMAs
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Shift jaws-teeth-lips as per Alligator definition (8,5,3 bars forward)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set rolled values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Elder Ray: EMA13 and power calculations
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # 1d HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 50)  # Need Alligator, volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation (rising bull power / falling bear power)
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # Long: Alligator up + Elder Ray bullish + volume + price > 1d EMA50
            if (alligator_long and 
                bull_power[i] > 0 and 
                bull_power_rising and 
                volume_confirm[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator down + Elder Ray bearish + volume + price < 1d EMA50
            elif (alligator_short and 
                  bear_power[i] < 0 and 
                  bear_power_falling and 
                  volume_confirm[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator reversal OR Elder Ray divergence
            if not alligator_long or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator reversal OR Elder Ray divergence
            if not alligator_short or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Volume_1dEMA50"
timeframe = "12h"
leverage = 1.0