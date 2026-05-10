#!/usr/bin/env python3
# 6H_MarketStructure_Reversal
# Hypothesis: Identify reversals at key market structure levels (swing highs/lows) with volume confirmation.
# Uses 1d swing points and 6h price action to catch institutional reversals.
# Works in bull/bear by trading mean reversion at extremes with volume confirmation.
# Target: 25-35 trades/year per symbol.

name = "6H_MarketStructure_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Volume moving average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # RSI (14-period) for momentum exhaustion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily swing points (structure levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Swing high: higher high followed by lower high
    swing_high = np.zeros_like(high_1d, dtype=bool)
    swing_low = np.zeros_like(low_1d, dtype=bool)
    
    for i in range(2, len(high_1d)-2):
        # Swing high: current high > previous 2 highs and next 2 highs
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = True
        # Swing low: current low < previous 2 lows and next 2 lows
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = True
    
    # Create arrays of swing levels (0 when no swing, price level when swing)
    swing_high_levels = np.where(swing_high, high_1d, 0)
    swing_low_levels = np.where(swing_low, low_1d, 0)
    
    # Forward fill swing levels to create resistance/support zones
    swing_high_levels = pd.Series(swing_high_levels).replace(0, np.nan).ffill().bfill().values
    swing_low_levels = pd.Series(swing_low_levels).replace(0, np.nan).ffill().bfill().values
    
    # Align swing levels to 6h
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high_levels)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low_levels)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.8
        
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Distance to swing levels (as percentage of price)
        dist_to_resistance = (swing_high_aligned[i] - close[i]) / close[i] if swing_high_aligned[i] > 0 else 1
        dist_to_support = (close[i] - swing_low_aligned[i]) / close[i] if swing_low_aligned[i] > 0 else 1
        
        near_resistance = dist_to_resistance < 0.005  # Within 0.5% of resistance
        near_support = dist_to_support < 0.005       # Within 0.5% of support
        
        if position == 0:
            # Enter long: near support, RSI oversold, volume confirmation
            if near_support and rsi_oversold and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: near resistance, RSI overbought, volume confirmation
            elif near_resistance and rsi_overbought and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 or price approaches resistance
            if rsi[i] > 50 or near_resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 or price approaches support
            if rsi[i] < 50 or near_support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals