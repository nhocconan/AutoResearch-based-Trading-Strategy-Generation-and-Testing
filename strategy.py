# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action relative to 1-day Volume-Weighted Average Price (VWAP)
# combined with 6h RSI(14) for momentum and 6h volume spike filter.
# VWAP acts as dynamic support/resistance: price above VWAP = bullish bias,
# price below VWAP = bearish bias. RSI filters for overbought/oversold conditions
# to avoid chasing extremes. Volume spike confirms institutional interest.
# Designed for 6h timeframe to capture institutional flows with low trade frequency.
# Works in both bull and bear markets by adapting to VWAP as dynamic fair value.
# Entry: Price crosses above VWAP with RSI<70 and volume spike (long).
# Entry: Price crosses below VWAP with RSI>30 and volume spike (short).
# Exit: Price crosses back across VWAP in opposite direction.
# Target: 15-30 trades/year (~60-120 over 4 years) to minimize fee drag.

name = "6h_VWAP_RSI_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP for each 6h bar using intraday approximation
    # VWAP = sum(price * volume) / sum(volume) where price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative sums for VWAP calculation (reset daily)
    # We'll use rolling window of 4 bars (24h/6h) to approximate daily VWAP
    vwap = pd.Series(vwap_numerator).rolling(window=4, min_periods=1).sum().values / \
           pd.Series(vwap_denominator).rolling(window=4, min_periods=1).sum().values
    
    # RSI(14) on 6h timeframe
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP + RSI not overbought + volume spike
            if (close[i] > vwap[i] and close[i-1] <= vwap[i-1] and  # crossed above
                rsi[i] < 70 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP + RSI not oversold + volume spike
            elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1] and  # crossed below
                  rsi[i] > 30 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses back below VWAP
            if close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses back above VWAP
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals