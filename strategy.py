#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Breakout_VolumeConfirmation_V2
Hypothesis: Breakouts above daily R1 or below daily S1 on 6h timeframe with volume confirmation and 6h momentum (price > 6h EMA20) yield high-probability trades. Uses tighter volume filter (1.8x average) and requires momentum alignment to reduce false breakouts. Targets 15-30 trades/year by requiring confluence of price breakout, volume surge, and trend alignment. Works in bull/bear markets by only taking breakouts in direction of 6h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA20 for momentum filter
    close_6h = prices['close'].values
    ema_20 = calculate_ema(close_6h, 20)
    
    # Align daily OHLC to 6h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily Camarilla levels from previous day's OHLC
        # Use prior bar's aligned daily values (previous completed day)
        prev_high = high_1d_aligned[i-1]
        prev_low = low_1d_aligned[i-1]
        prev_close = close_1d_aligned[i-1]
        
        r1, s1 = calculate_camarilla(prev_high, prev_low, prev_close)
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average (tighter than before)
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.8 * vol_ma
        else:
            volume_ok = False
        
        # Momentum filter: price above/below 6h EMA20
        price_above_ema = price > ema_20[i]
        price_below_ema = price < ema_20[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + price above EMA20
            if price > r1 and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + price below EMA20
            elif price < s1 and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or price falls below EMA20
            if price < s1 or price < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or price rises above EMA20
            if price > r1 or price > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Breakout_VolumeConfirmation_V2"
timeframe = "6h"
leverage = 1.0