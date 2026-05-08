#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action near 12h/1d volume-weighted average price (VWAP) with trend filter
# Uses VWAP as dynamic support/resistance in ranging markets and breakout confirmation in trends.
# Combines with 12h EMA50 trend filter and volume confirmation for high-probability entries.
# Designed for low-frequency trades (<120 total) to minimize fee drag in ranging/choppy markets.

name = "4h_VWAP_Bounce_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h VWAP (volume-weighted average price)
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    vwap_12h = (typical_price_12h * df_12h['volume'].values).cumsum() / df_12h['volume'].values.cumsum()
    vwap_12h = np.where(df_12h['volume'].values.cumsum() > 0, vwap_12h, np.nan)
    
    # Align 12h VWAP to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Get 12h data for EMA50 trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches VWAP support in uptrend with volume confirmation
            if (close[i] >= vwap_12h_aligned[i] * 0.998 and  # Near VWAP support (0.2% tolerance)
                close[i] > ema50_12h_aligned[i] and          # Above 12h EMA50 (uptrend)
                vol_confirm[i]):                           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Enter short: price touches VWAP resistance in downtrend with volume confirmation
            elif (close[i] <= vwap_12h_aligned[i] * 1.002 and  # Near VWAP resistance (0.2% tolerance)
                  close[i] < ema50_12h_aligned[i] and          # Below 12h EMA50 (downtrend)
                  vol_confirm[i]):                           # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below VWAP or trend fails
            if (close[i] < vwap_12h_aligned[i] * 0.995 or   # Below VWAP support
                close[i] < ema50_12h_aligned[i]):           # Trend failed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above VWAP or trend fails
            if (close[i] > vwap_12h_aligned[i] * 1.005 or   # Above VWAP resistance
                close[i] > ema50_12h_aligned[i]):           # Trend failed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals