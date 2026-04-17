#!/usr/bin/env python3
"""
4h Williams Alligator with 1d Trend Filter and Volume Confirmation
Long: Price > Alligator Teeth (red line) AND price > Alligator Lips (green line) AND Alligator Jaws < Teeth < Lips (bullish alignment) AND volume > 1.5x 4h volume SMA(20)
Short: Price < Alligator Teeth AND price < Alligator Lips AND Alligator Jaws > Teeth > Lips (bearish alignment) AND volume > 1.5x 4h volume SMA(20)
Exit: Price crosses back below/above Alligator Teeth OR Alligator alignment breaks
Uses Williams Alligator (13,8,5 SMAs) for trend, price/teeth relationship for entry, volume for confirmation
Target: 20-40 trades/year per symbol (80-160 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Williams Alligator on 4h: Jaws (13-bar, 8 offset), Teeth (8-bar, 5 offset), Lips (5-bar, 3 offset)
    close_4h = df_4h['close'].values
    sma_13 = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    sma_8 = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    sma_5 = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    
    jaws = np.roll(sma_13, 8)  # 8-period offset
    teeth = np.roll(sma_8, 5)   # 5-period offset
    lips = np.roll(sma_5, 3)    # 3-period offset
    
    # Align to 4h timeframe
    jaws_4h_aligned = align_htf_to_ltf(prices, df_4h, jaws)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Get 1d data for trend filter (close > EMA50)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 20)  # Ensure indicators are warmed up
    
    for i in range(start_idx, n):
        if (np.isnan(jaws_4h_aligned[i]) or np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        jaws_val = jaws_4h_aligned[i]
        teeth_val = teeth_4h_aligned[i]
        lips_val = lips_4h_aligned[i]
        ema_1d_val = ema_50_1d_aligned[i]
        
        # Bullish alignment: Jaws < Teeth < Lips
        bullish_aligned = jaws_val < teeth_val < lips_val
        # Bearish alignment: Jaws > Teeth > Lips
        bearish_aligned = jaws_val > teeth_val > lips_val
        
        if position == 0:
            # Long: Price > Teeth AND Price > Lips AND bullish alignment AND volume > 1.5x SMA AND price > 1d EMA50
            if (price > teeth_val and price > lips_val and bullish_aligned and 
                vol > 1.5 * vol_sma_val and price > ema_1d_val):
                signals[i] = 0.25
                position = 1
            # Short: Price < Teeth AND Price < Lips AND bearish alignment AND volume > 1.5x SMA AND price < 1d EMA50
            elif (price < teeth_val and price < lips_val and bearish_aligned and 
                  vol > 1.5 * vol_sma_val and price < ema_1d_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Teeth OR alignment breaks
            if price < teeth_val or not bullish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Teeth OR alignment breaks
            if price > teeth_val or not bearish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0