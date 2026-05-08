#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsAlligator_1dTrend_Volume"
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
    
    # Get daily data for trend and alligator calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components on daily timeframe
    # Jaw (Blue): 13-period SMMA, 8 bars forward
    # Teeth (Red): 8-period SMMA, 5 bars forward  
    # Lips (Green): 5-period SMMA, 3 bars forward
    # Using simple moving average with forward shift for simplicity
    
    # Jaw: 13-period SMA shifted 8 bars forward
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars forward  
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars forward
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Daily trend filter: EMA(34) on close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or 
            np.isnan(ema_34_1d_6h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals:
        # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
        # Bearish alignment: Lips < Teeth < Jaw (green < red < blue)
        bullish_align = (lips_6h[i] > teeth_6h[i]) and (teeth_6h[i] > jaw_6h[i])
        bearish_align = (lips_6h[i] < teeth_6h[i]) and (teeth_6h[i] < jaw_6h[i])
        
        if position == 0:
            # Long: Bullish alligator alignment + above daily EMA34 + volume confirmation
            if (bullish_align and
                close[i] > ema_34_1d_6h[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alligator alignment + below daily EMA34 + volume confirmation
            elif (bearish_align and
                  close[i] < ema_34_1d_6h[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish alligator alignment or price below EMA34
            if (bearish_align or
                close[i] < ema_34_1d_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alligator alignment or price above EMA34
            if (bullish_align or
                close[i] > ema_34_1d_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals