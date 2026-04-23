#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 Trend Filter and Volume Spike
- Long: Jaw < Teeth < Lips (Alligator bullish alignment) + price > 1d EMA50 (uptrend) + volume > 1.8x 20-period average
- Short: Jaw > Teeth > Lips (Alligator bearish alignment) + price < 1d EMA50 (downtrend) + volume > 1.8x 20-period average
- Exit: Alligator alignment reverses (Teeth crosses Jaw) or price crosses Teeth
- Uses Williams Alligator (SMAs with specific offsets) for trend identification and 1d EMA50 for higher timeframe trend filter
- Volume spike confirms institutional participation
- Discrete position sizing (0.25) to minimize fee churn
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Williams Alligator is effective in both trending and ranging markets by showing convergence/divergence
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator components (using 4h data)
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    # SMMA (Smoothed Moving Average) approximation using EMA with specific period
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max shift + longest period)
    start_idx = max(13 + 8, 8 + 5, 5 + 3, 50, 20)  # 21, 13, 8, 50, 20 -> 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Williams Alligator signals with trend filter and volume confirmation
        # Bullish alignment: Jaw < Teeth < Lips (Alligator waking up, eating)
        # Bearish alignment: Jaw > Teeth > Lips (Alligator sleeping)
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Volume confirmation
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        # Long: Bullish alignment + uptrend + volume spike
        # Short: Bearish alignment + downtrend + volume spike
        long_signal = bullish_alignment and uptrend and volume_spike
        short_signal = bearish_alignment and downtrend and volume_spike
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator alignment reverses or price crosses Teeth
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish alignment forms OR price crosses below Teeth
                if bearish_alignment or (close[i] < teeth[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Bullish alignment forms OR price crosses above Teeth
                if bullish_alignment or (close[i] > teeth[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0