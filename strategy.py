#!/usr/bin/env python3
"""
6h_Liquidity_Area_Reversal_with_OrderFlow
Hypothesis: In 6h timeframe, price often reverses at liquidity areas (equal highs/lows) when order flow shows exhaustion. 
Look for price touching recent equal highs/lows (within 0.5%) with declining volume (signaling exhaustion) and RSI divergence.
Works in both bull (selling exhaustion at resistance) and bear (buying exhaustion at support) markets.
Target: 20-40 trades/year on 6f to stay under fee drag limits.
"""

name = "6h_Liquidity_Area_Reversal_with_OrderFlow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) for divergence detection
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume moving average for exhaustion detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Lookback period for finding equal highs/lows
    lookback = 20
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(rsi_values[i-1]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Find recent equal highs and lows (within 0.5% tolerance)
        recent_highs = high[i-lookback:i]
        recent_lows = low[i-lookback:i]
        
        # Current price levels
        curr_high = high[i]
        curr_low = low[i]
        
        # Check for equal high (resistance level)
        equal_high = False
        if len(recent_highs) > 0:
            max_recent = np.max(recent_highs)
            if abs(curr_high - max_recent) / max_recent < 0.005:  # Within 0.5%
                equal_high = True
        
        # Check for equal low (support level)
        equal_low = False
        if len(recent_lows) > 0:
            min_recent = np.min(recent_lows)
            if abs(curr_low - min_recent) / min_recent < 0.005:  # Within 0.5%
                equal_low = True
        
        # Volume exhaustion: current volume < 70% of 20-period average
        vol_exhaustion = volume[i] < 0.7 * vol_ma_20[i]
        
        # RSI conditions for divergence
        rsi_overbought = rsi_values[i] > 70
        rsi_oversold = rsi_values[i] < 30
        rsi_bearish_div = (rsi_values[i] < rsi_values[i-1]) and rsi_values[i] > 60
        rsi_bullish_div = (rsi_values[i] > rsi_values[i-1]) and rsi_values[i] < 40
        
        if position == 0:
            # LONG setup: price at equal low + bullish RSI divergence + volume exhaustion
            if equal_low and rsi_bullish_div and vol_exhaustion:
                signals[i] = 0.25
                position = 1
            # SHORT setup: price at equal high + bearish RSI divergence + volume exhaustion
            elif equal_high and rsi_bearish_div and vol_exhaustion:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or volume surges (distribution)
            if rsi_values[i] > 70 or volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or volume surges (accumulation)
            if rsi_values[i] < 30 or volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals