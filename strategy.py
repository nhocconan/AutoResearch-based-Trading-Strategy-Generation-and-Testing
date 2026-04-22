#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume spike confirmation
# Long when price > Alligator Jaw + Bullish alignment + volume spike
# Short when price < Alligator Jaw + Bearish alignment + volume spike
# Exit when price crosses back below/above Jaw or trend reverses
# Designed for low trade frequency (~15-25/year) with strong trend-following edge in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Williams Alligator and trend filter
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Williams Alligator (13,8,5 smoothed with 8,5,3)
    # Jaw (13-period SMMA, smoothed by 8)
    sma13 = pd.Series(close_daily).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(sma13).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (8-period SMMA, smoothed by 5)
    sma8 = pd.Series(close_daily).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(sma8).rolling(window=5, min_periods=5).mean().values
    
    # Lips (5-period SMMA, smoothed by 3)
    sma5 = pd.Series(close_daily).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(sma5).rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_daily, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_daily, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_daily, lips)
    
    # Calculate daily EMA50 for trend filter
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        # Alligator alignment
        bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_alignment = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        if position == 0:
            # Long conditions: price > Jaw + Bullish alignment + volume spike
            if price > jaw_val and bullish_alignment and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw + Bearish alignment + volume spike
            elif price < jaw_val and bearish_alignment and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses Jaw or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below Jaw or trend turns down
                if price <= jaw_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above Jaw or trend turns up
                if price >= jaw_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0