#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# Price above all three lines = bullish trend, below all three = bearish trend.
# Uses 1d EMA50 for higher timeframe trend confirmation and volume spike for entry confirmation.
# Designed for 12h timeframe to capture medium-term trends with lower trade frequency.
# Works in both bull and bear markets by following trend direction.
# Uses discrete position sizing (0.25) to balance return and risk while minimizing fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().shift(8)
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().shift(5)
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().shift(3)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw.iloc[i]
        teeth_val = teeth.iloc[i]
        lips_val = lips.iloc[i]
        
        if position == 0:
            # Long: Price above all Alligator lines + above 1d EMA + volume spike
            if (close[i] > jaw_val and close[i] > teeth_val and close[i] > lips_val and
                close[i] > ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below all Alligator lines + below 1d EMA + volume spike
            elif (close[i] < jaw_val and close[i] < teeth_val and close[i] < lips_val and
                  close[i] < ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses any Alligator line in opposite direction
            if position == 1:
                # Exit long: Price closes below Jaw (weaker condition) or Teeth
                if close[i] < jaw_val or close[i] < teeth_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above Jaw (weaker condition) or Teeth
                if close[i] > jaw_val or close[i] > teeth_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0