#!/usr/bin/env python3
"""
6h Donchian breakout with 1d Supertrend filter and volume confirmation.
Hypothesis: Donchian breakouts aligned with 1d Supertrend direction capture strong trends.
Volume filter ensures momentum, reducing false breakouts. Works in bull (long breakouts) 
and bear (short breakdowns) via Supertrend regime filter. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14251_6h_donchian20_1d_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend with proper min_periods"""
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize final bands
    final_upper = np.full_like(upper_band, np.nan)
    final_lower = np.full_like(lower_band, np.nan)
    supertrend = np.full_like(close, np.nan)
    
    for i in range(1, len(close)):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(atr[i]):
            continue
            
        # Final upper band
        if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # Final lower band
        if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Supertrend
        if i == 1:
            supertrend[i] = final_upper[i]
        else:
            if supertrend[i-1] == final_upper[i-1] and close[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
            elif supertrend[i-1] == final_upper[i-1] and close[i] > final_upper[i]:
                supertrend[i] = final_lower[i]
            elif supertrend[i-1] == final_lower[i-1] and close[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
            elif supertrend[i-1] == final_lower[i-1] and close[i] < final_lower[i]:
                supertrend[i] = final_upper[i]
    
    # Direction: 1 for uptrend (price above supertrend), -1 for downtrend
    direction = np.where(close > supertrend, 1, -1)
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Supertrend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Supertrend
    supertrend_1d, trend_1d = calculate_supertrend(high_1d, low_1d, close_1d, 10, 3.0)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR, 10 for Supertrend)
    start = max(20, 20, 14, 10) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(trend_1d_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with 1d Supertrend filter and volume
        # Long: break above upper band + 1d uptrend + volume
        # Short: break below lower band + 1d downtrend + volume
        breakout_long = (close[i] > highest_high[i-1]) and (trend_1d_aligned[i] == 1) and vol_filter[i]
        breakout_short = (close[i] < lowest_low[i-1]) and (trend_1d_aligned[i] == -1) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or breakdown of lower band
            if close[i] <= stop_price or close[i] < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or breakout of upper band
            if close[i] >= stop_price or close[i] > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals