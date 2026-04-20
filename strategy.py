#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily Williams Alligator components (Jaw, Teeth, Lips)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate weekly trend filter (EMA34)
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema34_val = ema34_1w_aligned[i]
        vol_val = volume_1d[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(ema34_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_align = lips_val > teeth_val > jaw_val
        bearish_align = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long: Bullish Alligator alignment + price above weekly EMA34 + volume above average
            if bullish_align and close_val > ema34_val and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + price below weekly EMA34 + volume above average
            elif bearish_align and close_val < ema34_val and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks bearish OR price crosses below weekly EMA34
            if bearish_align or close_val < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks bullish OR price crosses above weekly EMA34
            if bullish_align or close_val > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_Alligator_WeeklyEMA34_Volume
# Uses Williams Alligator (13,8,5) on daily timeframe for entry signals
# Weekly EMA34 as trend filter to avoid counter-trend trades
# Volume confirmation to ensure institutional participation
# Designed for 1d timeframe with ~15-25 trades/year
name = "1d_Alligator_WeeklyEMA34_Volume"
timeframe = "1d"
leverage = 1.0