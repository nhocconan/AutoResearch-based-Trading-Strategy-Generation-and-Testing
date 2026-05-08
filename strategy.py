#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA21 trend filter + volume confirmation
# Long when price > Alligator's Jaw (blue line), price > Teeth (red line), price > Lips (green line),
# with 1d EMA21 upward slope and volume > 1.5x 20-period average
# Short when price < Jaw, price < Teeth, price < Lips, with 1d EMA21 downward slope and volume confirmation
# Exit when price crosses any Alligator line or 1d trend reverses
# Williams Alligator identifies trend phases (sleeping, awakening, feeding) - effective in trending markets
# 1d EMA21 filters counter-trend noise, volume confirms momentum strength
# Targets 50-150 total trades over 4 years (12-37/year) for optimal fee drag

name = "6h_WilliamsAlligator_1dEMA21_Volume"
timeframe = "6h"
leverage = 1.0

def williams_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Williams Alligator: three SMAs shifted into the future"""
    # Calculate SMAs
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Shift forward (into the future) as per Williams Alligator specification
    jaw = np.roll(jaw, -jaw_period//2)
    teeth = np.roll(teeth, -teeth_period//2)
    lips = np.roll(lips, -lips_period//2)
    
    # Fill shifted values with NaN for the shifted periods
    jaw[:jaw_period//2] = np.nan
    teeth[:teeth_period//2] = np.nan
    lips[:lips_period//2] = np.nan
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h data
    jaw, teeth, lips = williams_alligator(high, low, close)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA21 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_slope = ema21_1d[1:] - ema21_1d[:-1]  # slope: positive = uptrend
    ema21_1d_slope = np.concatenate([[0], ema21_1d_slope])  # align length
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema21_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(ema21_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema21_val = ema21_1d_aligned[i]
        ema21_slope = ema21_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price above all Alligator lines, volume confirmation, 1d uptrend
            if (close_val > jaw_val and close_val > teeth_val and close_val > lips_val and
                vol_conf_val and ema21_slope > 0):
                signals[i] = 0.25
                position = 1
            # Enter short: price below all Alligator lines, volume confirmation, 1d downtrend
            elif (close_val < jaw_val and close_val < teeth_val and close_val < lips_val and
                  vol_conf_val and ema21_slope < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below any Alligator line or 1d trend turns down
            if (close_val < jaw_val or close_val < teeth_val or close_val < lips_val or ema21_slope < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above any Alligator line or 1d trend turns up
            if (close_val > jaw_val or close_val > teeth_val or close_val > lips_val or ema21_slope > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals