#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator trend filter with 1d support/resistance bounce and volume confirmation.
# In trending markets (jaw < teeth < lips for long, jaw > teeth > lips for short): trade pullbacks to the teeth (middle line).
# Uses 1d pivot points for dynamic support/resistance and volume > 1.5x 20-period average for confirmation.
# Target: 15-35 trades/year by requiring trend alignment + pullback to teeth + volume confirmation.
# Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift).

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length == 0:
        return source
    result = np.full_like(source, np.nan, dtype=float)
    sma = np.convolve(source, np.ones(length)/length, mode='valid')
    if len(sma) == 0:
        return result
    result[length-1:len(sma)+length-1] = sma
    for i in range(len(sma)+length-1, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator on close prices
    close = prices['close'].values
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Calculate 1d pivot points (standard floor method)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    support1 = (2 * pivot) - df_1d['high'].values
    resistance1 = (2 * pivot) - df_1d['low'].values
    
    # Align 1d pivot data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    support1_aligned = align_htf_to_ltf(prices, df_1d, support1)
    resistance1_aligned = align_htf_to_ltf(prices, df_1d, resistance1)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(support1_aligned[i]) or 
            np.isnan(resistance1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Williams Alligator trend detection
        # Bullish alignment: lips > teeth > jaw
        # Bearish alignment: jaw > teeth > lips
        is_bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        is_bearish = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        if position == 0:
            if volume_confirm:
                # Look for pullbacks to the teeth (middle line) in the direction of the trend
                teeth_val = teeth[i]
                # For long: price pulls back to or slightly above teeth in bullish alignment
                # For short: price pulls back to or slightly below teeth in bearish alignment
                if is_bullish and price >= teeth_val * 0.995 and price <= teeth_val * 1.005:
                    signals[i] = 0.25
                    position = 1
                elif is_bearish and price >= teeth_val * 0.995 and price <= teeth_val * 1.005:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: price moves away from teeth or trend changes
            exit_signal = False
            teeth_val = teeth[i]
            
            if position == 1:  # long position
                # Exit if trend turns bearish or price moves significantly below teeth
                if not is_bullish or price < teeth_val * 0.98:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit if trend turns bullish or price moves significantly above teeth
                if not is_bearish or price > teeth_val * 1.02:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_Pullback_Pivot_Volume"
timeframe = "12h"
leverage = 1.0