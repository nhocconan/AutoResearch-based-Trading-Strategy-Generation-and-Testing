#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Williams_Alligator_Trend_Follow_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator: 3 SMAs (Jaws, Teeth, Lips)
    # Jaws: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    close_1d = df_1d['close'].values
    
    sma5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    sma8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    
    # Shift forward (Alligator lines are shifted into the future)
    lips = np.roll(sma5, 3)
    teeth = np.roll(sma8, 5)
    jaws = np.roll(sma13, 8)
    
    # Set initial values to NaN due to shift
    lips[:3] = np.nan
    teeth[:5] = np.nan
    jaws[:8] = np.nan
    
    # Align Alligator lines to 6h timeframe
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    jaws_6h = align_htf_to_ltf(prices, df_1d, jaws)
    
    # Trend filter: 34 EMA on 1d
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: 20-period average volume spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(jaws_6h[i]) or 
            np.isnan(ema34_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Lips > Teeth > Jaws (bullish alignment) AND price above EMA34 AND volume spike
            if (lips_6h[i] > teeth_6h[i] > jaws_6h[i]) and (close[i] > ema34_6h[i]) and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) AND price below EMA34 AND volume spike
            elif (jaws_6h[i] > teeth_6h[i] > lips_6h[i]) and (close[i] < ema34_6h[i]) and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks down (lips < teeth OR teeth < jaws) OR price crosses below EMA34
            if (lips_6h[i] < teeth_6h[i]) or (teeth_6h[i] < jaws_6h[i]) or (close[i] < ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks down (jaws < teeth OR teeth < lips) OR price crosses above EMA34
            if (jaws_6h[i] < teeth_6h[i]) or (teeth_6h[i] < lips_6h[i]) or (close[i] > ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals