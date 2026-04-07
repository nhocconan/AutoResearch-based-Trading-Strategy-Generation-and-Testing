#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily pivot reversal logic
# Uses Donchian(20) channel for breakout direction and daily Camarilla pivot levels
# for mean reversion entries. Goes long on breakouts above daily R3 and short on
# breakdowns below daily S3, with continuation beyond R4/S4. Volume filter ensures
# momentum confirmation. Designed for low frequency (target: 15-35 trades/year)
# to minimize fee drift while capturing both trending and mean-reverting moves.

name = "6h_donchian20_daily_pivot_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channel
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    # Camarilla levels: C = close, range = high - low
    # R4 = C + (range * 1.1/2), R3 = C + (range * 1.1/4), R2 = C + (range * 1.1/6), R1 = C + (range * 1.1/12)
    # S1 = C - (range * 1.1/12), S2 = C - (range * 1.1/6), S3 = C - (range * 1.1/4), S4 = C - (range * 1.1/2)
    camarilla_r4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align daily Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation
        bullish_breakout = (close[i] > highest_high[i]) and vol_filter[i]
        bearish_breakout = (close[i] < lowest_low[i]) and vol_filter[i]
        
        # Mean reversion conditions at extreme pivot levels
        bullish_reversal = (close[i] <= s3_6h[i]) and vol_filter[i]
        bearish_reversal = (close[i] >= r3_6h[i]) and vol_filter[i]
        
        # Strong continuation beyond R4/S4
        strong_bullish = (close[i] > r4_6h[i]) and vol_filter[i]
        strong_bearish = (close[i] < s4_6h[i]) and vol_filter[i]
        
        # Generate signals
        if bullish_breakout or strong_bullish or bullish_reversal:
            signals[i] = 0.25
        elif bearish_breakout or strong_bearish or bearish_reversal:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals