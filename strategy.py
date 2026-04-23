#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA8 trend filter and volume confirmation.
- Williams Alligator: Jaw (EMA13, 8-period shift), Teeth (EMA8, 5-period shift), Lips (EMA5, 3-period shift)
- Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA8 AND volume > 1.5x 20-period avg
- Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA8 AND volume > 1.5x 20-period avg
- Exit: Alligator alignment reverses OR price crosses 1w EMA8
- Works in bull (buy strength on dips) and bear (sell weakness on rallies)
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator components
    # Jaw: EMA13, 8-period shift
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: EMA8, 5-period shift
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: EMA5, 3-period shift
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Calculate 1w EMA8 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 8, 5, 20)  # Need 50 for safety, 13 for jaw, 8 for teeth, 5 for lips, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(lips[i]) or
            np.isnan(teeth[i]) or
            np.isnan(jaw[i]) or
            np.isnan(ema_8_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Williams Alligator signals
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish alignment AND price > 1w EMA8
            if (bullish_alignment and 
                volume_confirm and 
                close[i] > ema_8_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price < 1w EMA8
            elif (bearish_alignment and 
                  volume_confirm and 
                  close[i] < ema_8_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish alignment OR price < 1w EMA8 (trend flip)
            if bearish_alignment or close[i] < ema_8_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alignment OR price > 1w EMA8 (trend flip)
            if bullish_alignment or close[i] > ema_8_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0