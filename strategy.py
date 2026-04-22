# 1. Hypothesis: 4-hour Williams Alligator with 12-hour Trend and Volume Confirmation
# Long when Alligator lines are bullish (Green > Red > Blue) and 12h EMA50 rising with volume spike.
# Short when Alligator lines are bearish (Blue > Red > Green) and 12h EMA50 falling with volume spike.
# Exit when Alligator lines cross or 12h EMA50 reverses.
# Designed for low trade frequency by requiring multiple confirmations.
# Works in both bull and bear markets by following the 12h trend.

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
    
    # Williams Alligator: Smoothed Moving Average (SMA-like) with different periods and shifts
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    # SMMA = (Previous SMMA * (period-1) + Current Close) / period
    
    def smma(series, period):
        result = np.full_like(series, np.nan, dtype=float)
        if len(series) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (Previous SMMA * (period-1) + Current Close) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line  
    lips = smma(close, 5)   # Green line
    
    # Shift the lines forward (Williams Alligator shifts)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate the shifted values at the beginning
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bullish: Lips > Teeth > Jaw (Green > Red > Blue)
            bullish = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
            # Bearish: Jaw > Teeth > Lips (Blue > Red > Green)
            bearish = jaw_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i]
            
            # Long: Bullish Alligator, 12h EMA50 rising, volume spike
            if (bullish and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator, 12h EMA50 falling, volume spike
            elif (bearish and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross or 12h EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Bullish alignment breaks or 12h EMA50 turns down
                bullish = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
                if not bullish or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bearish alignment breaks or 12h EMA50 turns up
                bearish = jaw_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i]
                if not bearish or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0