#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with volume confirmation and 1d EMA50 trend filter.
# Long when green line > red line > blue line (bullish alignment) + volume > 1.5x average + price > 1d EMA50
# Short when green line < red line < blue line (bearish alignment) + volume > 1.5x average + price < 1d EMA50
# Exit when alignment breaks or volume drops below 0.8x average.
# Williams Alligator uses SMAs: Jaw (13,8), Teeth (8,5), Lips (5,3).
# Designed to catch trends in both bull and bear markets while avoiding chop.
# Target: 15-25 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator lines (all based on median price)
    # Median price = (high + low) / 2
    median_price = (high_12h + low_12h) / 2
    
    # Jaw: Blue line - 13-period SMA, smoothed by 8 periods
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean()
    
    # Teeth: Red line - 8-period SMA, smoothed by 5 periods
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean()
    
    # Lips: Green line - 5-period SMA, smoothed by 3 periods
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean()
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_vals)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter (24-period average on 12h timeframe)
    volume = prices['volume'].values
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 24-period average
        vol_filter = vol > 1.5 * vol_ma
        
        # Williams Alligator alignment
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: bullish alignment + volume filter + price > EMA50
            if bullish_alignment and vol_filter and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + volume filter + price < EMA50
            elif bearish_alignment and vol_filter and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: alignment breaks or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or volume drops
                if not bullish_alignment or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or volume drops
                if not bearish_alignment or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_Volume_EMA50"
timeframe = "12h"
leverage = 1.0