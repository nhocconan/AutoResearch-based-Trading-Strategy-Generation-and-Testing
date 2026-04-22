#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with Fractal Filter + Volume Spike + Daily Trend Filter.
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and strength.
# Adds Williams Fractals for entry confirmation and daily EMA for trend filter.
# Designed to work in both bull and bear markets by trading with the trend only.
# Targets 20-40 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 4h data
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA, shifted 3 bars)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Calculate Williams Fractals on 4h data
    high = prices['high'].values
    low = prices['low'].values
    
    bearish_fractal = np.zeros(n)
    bullish_fractal = np.zeros(n)
    
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest of high[i-2:i+3]
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = high[i]
        
        # Bullish fractal: low[i] is lowest of low[i-2:i+3]
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Align Fractals to ensure proper timing (no look-ahead)
    bearish_fractal_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price above/below daily EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # Alligator alignment: all lines in proper order
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Alligator sleeping (intertwined) - no trade
        alligator_sleeping = (abs(jaw_val - teeth_val) < (jaw_val * 0.001) and 
                             abs(teeth_val - lips_val) < (teeth_val * 0.001))
        
        # Alligator awake and eating: proper alignment
        # Bullish: Lips > Teeth > Jaw
        bullish_alignment = lips_val > teeth_val > jaw_val
        # Bearish: Jaw > Teeth > Lips
        bearish_alignment = jaw_val > teeth_val > lips_val
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Look for fractal confirmation with trend and Alligator alignment
            if (bullish_fractal_aligned[i] > 0 and uptrend and 
                bullish_alignment and not alligator_sleeping and vol_spike):
                signals[i] = 0.25
                position = 1
            elif (bearish_fractal_aligned[i] > 0 and downtrend and 
                  bearish_alignment and not alligator_sleeping and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on bearish fractal or Alligator death (Lips < Jaw)
                if (bearish_fractal_aligned[i] > 0 or lips_val < jaw_val):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on bullish fractal or Alligator death (Jaw < Lips)
                if (bullish_fractal_aligned[i] > 0 or jaw_val < lips_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Alligator_Fractal_TrendFilter"
timeframe = "4h"
leverage = 1.0