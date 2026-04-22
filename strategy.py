#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams Alligator with 1-day EMA trend filter and volume confirmation.
Trades in the direction of the daily EMA trend when the Alligator lines are aligned and price crosses the Jaw line.
Uses volume spike to confirm institutional interest. Designed for low trade frequency (20-50 trades/year) to minimize
fee drag and work in both bull and bear markets by aligning with higher timeframe trend and using momentum at alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) SMAs of median price."""
    median = (high + low) / 2
    jaw = pd.Series(median).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median).rolling(window=5, min_periods=5).mean().values
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter and Alligator calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Alligator lines
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
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0 and vol_spike:
            # Long: price crosses above Jaw with bullish alignment and uptrend bias
            if close[i] > jaw_aligned[i] and bullish_alignment and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Jaw with bearish alignment and downtrend bias
            elif close[i] < jaw_aligned[i] and bearish_alignment and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite side of Jaw or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Jaw or closes below daily EMA
                if close[i] < jaw_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Jaw or closes above daily EMA
                if close[i] > jaw_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Alligator_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0