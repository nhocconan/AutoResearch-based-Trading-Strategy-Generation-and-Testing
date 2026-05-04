#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA(50) trend filter and volume spike confirmation
# Williams Alligator (Jaw=TEETH=13, Teeth=TEETH=8, Lips=TEETH=5) identifies trend absence/presence
# When all three lines are intertwined (chop), no trade; when aligned (trend), trade in direction
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.8x 24 EMA volume) filters false breakouts in low volatility
# Discrete sizing 0.25 minimizes fee churn targeting 50-150 total trades over 4 years (12-37/year)
# Works in bull markets (trend up: Lips>Teeth>Jaw) and bear markets (trend down: Lips<Teeth<Jaw)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_Balanced"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 12h timeframe (smoothed with 5,8,13 periods)
    # Jaw: Smoothed Median Price (13,8)
    # Teeth: Smoothed Median Price (8,5)
    # Lips: Smoothed Median Price (5,3)
    median_price = (high + low) / 2
    
    # Jaw (13,8)
    jaw = pd.Series(median_price).ewm(span=13, adjust=False).mean().ewm(span=8, adjust=False).mean().values
    # Teeth (8,5)
    teeth = pd.Series(median_price).ewm(span=8, adjust=False).mean().ewm(span=5, adjust=False).mean().values
    # Lips (5,3)
    lips = pd.Series(median_price).ewm(span=5, adjust=False).mean().ewm(span=3, adjust=False).mean().values
    
    # Shift to use prior completed bar (avoid look-ahead)
    jaw_shifted = np.roll(jaw, 1)
    teeth_shifted = np.roll(teeth, 1)
    lips_shifted = np.roll(lips, 1)
    jaw_shifted[0] = np.nan
    teeth_shifted[0] = np.nan
    lips_shifted[0] = np.nan
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Volume confirmation: 24-period EMA of volume (2*12h)
    vol_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume spike
            if lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_24[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume spike
            elif lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_24[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines intertwine (chop) OR price crosses below 1d EMA50
            if (lips_shifted[i] <= teeth_shifted[i] or teeth_shifted[i] <= jaw_shifted[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines intertwine (chop) OR price crosses above 1d EMA50
            if (lips_shifted[i] >= teeth_shifted[i] or teeth_shifted[i] >= jaw_shifted[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals