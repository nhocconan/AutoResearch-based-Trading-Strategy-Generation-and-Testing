#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Alligator_Fractal_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator and fractals (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Alligator components (13,8,5 SMAs with future shifts)
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw_1d = jaw_1d.shift(8)  # shift forward by 8 bars
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth_1d = teeth_1d.shift(5)  # shift forward by 5 bars
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips_1d = lips_1d.shift(3)  # shift forward by 3 bars
    
    # Williams Fractals (requires 2-bar confirmation)
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align Alligator lines to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d.values)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d.values)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d.values)
    
    # Align fractals with 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Align 1w close for trend filter
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(close_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        bearish_fract = bearish_fractal_aligned[i]
        bullish_fract = bullish_fractal_aligned[i]
        weekly_close = close_1w_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Alligator alignment: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
        alligator_bullish = lips > teeth and teeth > jaw
        alligator_bearish = lips < teeth and teeth < jaw
        
        if position == 0:
            # Long: bullish Alligator + price above weekly close + bullish fractal + volume
            if alligator_bullish and price > weekly_close and bullish_fract and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator + price below weekly close + bearish fractal + volume
            elif alligator_bearish and price < weekly_close and bearish_fract and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish Alligator alignment or price below weekly close
            if not alligator_bullish or price < weekly_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish Alligator alignment or price above weekly close
            if not alligator_bearish or price > weekly_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals