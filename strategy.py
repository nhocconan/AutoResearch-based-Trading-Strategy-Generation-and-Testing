#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1-day trend filter
# Uses Alligator (Jaw/Teeth/Lips) to detect trend and Elder Ray (Bull/Bear Power) for entry timing.
# Filters by 1-day EMA50 trend to avoid counter-trend trades.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 15-25 trades/year per symbol (60-100 total) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1-day close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Williams Alligator (13,8,5) on 6h data
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Smoothed median price (typical price)
    typical_price = (high + low + close) / 3.0
    
    # Jaw (13-period smoothed, shifted 8 bars)
    jaw_raw = pd.Series(typical_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth (8-period smoothed, shifted 5 bars)
    teeth_raw = pd.Series(typical_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips (5-period smoothed, shifted 3 bars)
    lips_raw = pd.Series(typical_price).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align 1-day EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Mouth open (Lips outside Teeth/Jaw) indicates trend
        # Lips above Teeth and Teeth above Jaw = uptrend
        # Lips below Teeth and Teeth below Jaw = downtrend
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Uptrend (Lips > Teeth > Jaw) + Bull Power > 0 + price above 1d EMA50
            if lips_above_teeth and teeth_above_jaw and bull_power[i] > 0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (Lips < Teeth < Jaw) + Bear Power < 0 + price below 1d EMA50
            elif lips_below_teeth and teeth_below_jaw and bear_power[i] < 0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Trend weakness (Lips crosses Teeth) or contrary Elder Ray signal
            if position == 1:
                if lips[i] < teeth[i] or bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > teeth[i] or bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Alligator_ElderRay_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0