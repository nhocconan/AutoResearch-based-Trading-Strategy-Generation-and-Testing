#!/usr/bin/env python3
"""
12h Williams Alligator with Volume Spike and EMA50 Trend Filter
Hypothesis: Williams Alligator identifies trend periods when jaws, teeth, and lips are aligned. 
Price trading outside the Alligator's mouth with volume confirmation indicates strong momentum.
In bull markets, buy when price > lips with bullish alignment; in bear markets, sell when price < teeth with bearish alignment.
Volume filter reduces false signals. Weekly trend filter ensures we trade with the higher timeframe momentum.
Designed for low-frequency trading (~20-40 trades/year) to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with future shift"""
    if len(close) < max(jaw_period, teeth_period, lips_period):
        return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)
    
    # Calculate SMAs
    jaw = np.full_like(close, np.nan)
    teeth = np.full_like(close, np.nan)
    lips = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if i >= jaw_period - 1:
            jaw[i] = np.mean(close[i - jaw_period + 1:i + 1])
        if i >= teeth_period - 1:
            teeth[i] = np.mean(close[i - teeth_period + 1:i + 1])
        if i >= lips_period - 1:
            lips[i] = np.mean(close[i - lips_period + 1:i + 1])
    
    # Williams Alligator shifts the lines forward: Jaw by 8, Teeth by 5, Lips by 3 bars
    # To avoid look-ahead, we use the unshifted values for signal generation
    # The alignment happens naturally through price relationship
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i < 49:
            ema_50_1w[i] = np.mean(close_1w[0:i+1])
        else:
            ema_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 12h data
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Volume confirmation: current volume > 2.0x 30-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 29:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-29:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment
        bullish_alignment = (lips[i] > teeth[i] > jaw[i])  # Lips > Teeth > Jaw
        bearish_alignment = (jaw[i] > teeth[i] > lips[i])  # Jaw > Teeth > Lips
        
        if position == 0:
            # Long: Price above Lips, bullish alignment, volume spike, and above weekly EMA50
            if (close[i] > lips[i] and bullish_alignment and vol_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below Teeth, bearish alignment, volume spike, and below weekly EMA50
            elif (close[i] < teeth[i] and bearish_alignment and vol_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Jaw or Alligator loses bullish alignment
            if close[i] < jaw[i] or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Teeth or Alligator loses bearish alignment
            if close[i] > teeth[i] or not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_VolumeSpike_EMA50Trend"
timeframe = "12h"
leverage = 1.0