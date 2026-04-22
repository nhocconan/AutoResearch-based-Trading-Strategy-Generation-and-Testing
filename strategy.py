#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator strategy with volume spike and daily trend filter.
# Uses Williams Alligator (jaw=13, teeth=8, lips=5 SMAs) to identify trend direction.
# Long when lips > teeth > jaw (bullish alignment) + volume spike + price > daily EMA50
# Short when lips < teeth < jaw (bearish alignment) + volume spike + price < daily EMA50
# Exit when Alligator lines cross (lips/teeth crossover) or volume drops below 70% of average.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Williams Alligator is effective in both trending and ranging markets when combined with volume and trend filters.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Williams Alligator calculation and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (using median price)
    median_price_1d = (high_1d + low_1d) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw_1d = np.roll(jaw_1d, 8)
    jaw_1d[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth_1d = np.roll(teeth_1d, 5)
    teeth_1d[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips_1d = np.roll(lips_1d, 3)
    lips_1d[:3] = np.nan
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (24-period average for 12h)
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
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 24-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Alligator alignment conditions
        bullish_alignment = lips > teeth and teeth > jaw
        bearish_alignment = lips < teeth and teeth < jaw
        
        if position == 0:
            # Long conditions: bullish alignment + volume spike + price > EMA50
            if bullish_alignment and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + volume spike + price < EMA50
            elif bearish_alignment and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator crossover (lips/teeth cross) or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when lips cross below teeth or volume dries up
                if lips < teeth or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when lips cross above teeth or volume dries up
                if lips > teeth or vol < 0.7 * vol_ma:
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