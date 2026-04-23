#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray + Weekly Pivot Direction
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
  Trend up: Lips > Teeth > Jaw; Trend down: Lips < Teeth < Jaw
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
  Long when Bull Power > 0 and rising; Short when Bear Power < 0 and falling
- Weekly Pivot: PP = (Prior week High + Low + Close)/3
  Bias: Long bias if price > PP, Short bias if price < PP
- Entry: Alligator trend + Elder Ray agreement + Weekly pivot bias
- Exit: Alligator trend reversal OR Elder Ray divergence
- Uses discrete sizing ±0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator SMAs
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13, 8 shift
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8, 5 shift
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5, 3 shift
    
    # Elder Ray: Bull/Bear Power vs EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # Get weekly HTF data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly OHLC for pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly Pivot Point: PP = (H + L + C)/3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (using prior week close for look-ahead safety)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # Need 13 for EMA13/Alligator, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator trend detection
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray strength and momentum
        bull_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        bear_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        
        if position == 0:
            # Long: Alligator up + Bull Power > 0 and rising + price > Weekly PP
            if (alligator_up and 
                bull_power[i] > 0 and bull_rising and 
                close[i] > weekly_pp_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator down + Bear Power < 0 and falling + price < Weekly PP
            elif (alligator_down and 
                  bear_power[i] < 0 and bear_falling and 
                  close[i] < weekly_pp_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator down OR Bear Power >= 0 (loss of bullish momentum)
            if alligator_down or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator up OR Bull Power <= 0 (loss of bearish momentum)
            if alligator_up or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_WeeklyPivot"
timeframe = "6h"
leverage = 1.0