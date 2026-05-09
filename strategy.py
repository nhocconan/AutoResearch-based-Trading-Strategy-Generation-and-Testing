#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Adaptive Supertrend with Weekly Trend Filter and Volume Spike
# Uses ATR-based dynamic bands that adapt to volatility, effective in trending and ranging markets.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation.
# Designed for low trade frequency (15-35/year) to minimize fee drag.
name = "6h_AdaptiveSupertrend_WeeklyTrend_VolumeSpike"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize final bands
    final_upper = np.full_like(close, np.nan)
    final_lower = np.full_like(close, np.nan)
    
    for i in range(1, len(close)):
        if close[i-1] > final_upper[i-1]:
            final_upper[i] = max(upper_band[i], final_upper[i-1])
        else:
            final_upper[i] = upper_band[i]
            
        if close[i-1] < final_lower[i-1]:
            final_lower[i] = min(lower_band[i], final_lower[i-1])
        else:
            final_lower[i] = lower_band[i]
    
    # Determine Supertrend direction
    supertrend = np.full_like(close, np.nan)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close)):
        if np.isnan(final_upper[i]) or np.isnan(final_lower[i]):
            continue
        if i == 0:
            supertrend[i] = final_lower[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] <= final_upper[i]:
                    supertrend[i] = final_lower[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_upper[i]
                    direction[i] = 1
            else:
                if close[i] >= final_lower[i]:
                    supertrend[i] = final_upper[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_lower[i]
                    direction[i] = -1
    
    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_6h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 200)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(direction[i]) or 
            np.isnan(ema200_1w_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Supertrend uptrend + price above weekly EMA200 + volume spike
            if direction[i] > 0 and close[i] > ema200_1w_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend + price below weekly EMA200 + volume spike
            elif direction[i] < 0 and close[i] < ema200_1w_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend turns downtrend OR price below weekly EMA200
            if direction[i] < 0 or close[i] < ema200_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend turns uptrend OR price above weekly EMA200
            if direction[i] > 0 or close[i] > ema200_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals