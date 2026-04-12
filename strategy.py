#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_ema_crossover_with_volume_filter
# Uses 4h EMA crossover (fast/slow) for trend direction, 1h for entry timing with volume confirmation.
# Long when 4h fast EMA > slow EMA, price > 1h VWAP, and volume > 1.5x 20-period average.
# Short when 4h fast EMA < slow EMA, price < 1h VWAP, and volume > 1.5x 20-period average.
# Exit when 4h EMA crossover reverses or price crosses 1h VWAP in opposite direction.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drift.
# Works in trending markets via EMA crossover and avoids false signals with volume filter.

name = "1h_4h_ema_crossover_with_volume_filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(9) and EMA(21)
    close_4h = df_4h['close'].values
    ema_fast = pd.Series(close_4h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_slow)
    
    # Calculate 1h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).cumsum().values
    vwap_denominator = pd.Series(volume).cumsum().values
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, 0.0)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check for 4h EMA crossover
        ema_bullish = ema_fast_aligned[i] > ema_slow_aligned[i]
        ema_bearish = ema_fast_aligned[i] < ema_slow_aligned[i]
        
        # Long conditions: 4h EMA bullish, price > VWAP, volume confirmation
        if ema_bullish and close[i] > vwap[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.20
        # Short conditions: 4h EMA bearish, price < VWAP, volume confirmation
        elif ema_bearish and close[i] < vwap[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: EMA crossover reverses or price crosses VWAP in opposite direction
        elif position == 1 and (not ema_bullish or close[i] < vwap[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not ema_bearish or close[i] > vwap[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals