#/usr/bin/env python3

"""
Hypothesis: 12-hour Williams Alligator with 1-day EMA trend filter and volume confirmation.
Trades when the Alligator lines (jaw, teeth, lips) align in bullish/bearish order and price is outside the mouth,
in the direction of the daily EMA trend. Volume spike confirms institutional interest. Designed for low trade
frequency (12-37 trades/year) to minimize fee drift and work in both bull and bear markets by combining
trend-following with volatility-based entry filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """
    Calculate Williams Alligator lines:
    Jaw (blue): 13-period SMMA, shifted 8 bars forward
    Teeth (red): 8-period SMMA, shifted 5 bars forward
    Lips (green): 5-period SMMA, shifted 3 bars forward
    SMMA (Smoothed Moving Average) is similar to EMA but with different smoothing.
    We'll use EMA as a proxy for SMMA for simplicity and speed.
    """
    # Use EMA as proxy for SMMA
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean()
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean()
    lips = pd.Series(close).ewm(span=5, adjust=False).mean()
    
    # Shift forward: jaw 8, teeth 5, lips 3
    jaw_shifted = jaw.shift(8)
    teeth_shifted = teeth.shift(5)
    lips_shifted = lips.shift(3)
    
    return jaw_shifted.values, teeth_shifted.values, lips_shifted.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter and Alligator calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Alligator lines (jaw, teeth, lips)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    jaw, teeth, lips = calculate_alligator(high_1d, low_1d, close_1d_arr)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Bullish alignment: Lips > Teeth > Jaw and price above Lips
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > lips_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Bearish alignment: Lips < Teeth < Jaw and price below Lips
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < lips_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns inside the Alligator's mouth (between teeth and jaw) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below teeth or trend turns bearish
                if close[i] < teeth_aligned[i] or lips_aligned[i] < teeth_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price closes above teeth or trend turns bullish
                if close[i] > teeth_aligned[i] or lips_aligned[i] > teeth_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0