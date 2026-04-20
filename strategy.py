#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for indicator calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for Williams %R
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R(14) calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (with 1-day delay for signal confirmation)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r, additional_delay_bars=1)
    
    # 6h price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 60-period (5-day) high/low for breakout levels
    highest_high_60 = pd.Series(high).rolling(window=60, min_periods=60).max().values
    lowest_low_60 = pd.Series(low).rolling(window=60, min_periods=60).min().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in critical values
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(highest_high_60[i]) or np.isnan(lowest_low_60[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        williams_r_val = williams_r_aligned[i]
        
        # Breakout conditions with volume confirmation
        bullish_breakout = (price > highest_high_60[i]) and (vol_ratio > 1.5)
        bearish_breakout = (price < lowest_low_60[i]) and (vol_ratio > 1.5)
        
        # Williams %R conditions for mean reversion in ranging markets
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        
        if position == 0:
            # Enter long on bullish breakout with volume
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish breakout with volume
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
            # Mean reversion entries in ranging markets
            elif oversold and (price > lowest_low_60[i]):  # Avoid catching falling knives
                signals[i] = 0.20
                position = 1
            elif overbought and (price < highest_high_60[i]):  # Avoid catching rising tops
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout or overbought conditions
            if bearish_breakout or overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout or oversold conditions
            if bullish_breakout or oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Breakout_MeanReversion_V1"
timeframe = "6h"
leverage = 1.0