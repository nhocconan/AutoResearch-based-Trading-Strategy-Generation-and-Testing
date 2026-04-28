#!/usr/bin/env python3
"""
1d_RSI_Divergence_BullBear_Filter
Hypothesis: Daily RSI with bull/bear divergence detection and volume confirmation captures trend reversals in both bull and bear markets. Bullish divergence (price makes lower low, RSI makes higher low) signals long; bearish divergence (price makes higher high, RSI makes lower high) signals short. Uses 14-period RSI with volume confirmation to filter false signals. Target: 10-20 trades/year per symbol to minimize fee drift while capturing major reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 14-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for RSI to stabilize
    
    for i in range(start_idx, n):
        # Skip if RSI or volume data is not ready
        if np.isnan(rsi[i]) or np.isnan(volume_confirmed[i]):
            signals[i] = 0.0
            continue
        
        # Look back for divergence signals (need at least 5 periods back)
        if i < 5:
            signals[i] = 0.0
            continue
            
        # Bullish divergence: price makes lower low, RSI makes higher low
        price_lower_low = low[i] < low[i-5] and low[i] == np.min(low[i-5:i+1])
        rsi_higher_low = rsi[i] > rsi[i-5] and rsi[i] == np.max(rsi[i-5:i+1])
        bullish_divergence = price_lower_low and rsi_higher_low
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        price_higher_high = high[i] > high[i-5] and high[i] == np.max(high[i-5:i+1])
        rsi_lower_high = rsi[i] < rsi[i-5] and rsi[i] == np.min(rsi[i-5:i+1])
        bearish_divergence = price_higher_high and rsi_lower_high
        
        # Entry conditions with volume confirmation
        long_entry = bullish_divergence and volume_confirmed[i] and rsi[i] < 40
        short_entry = bearish_divergence and volume_confirmed[i] and rsi[i] > 60
        
        # Exit on opposite divergence
        long_exit = bearish_divergence and volume_confirmed[i]
        short_exit = bullish_divergence and volume_confirmed[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_RSI_Divergence_BullBear_Filter"
timeframe = "1d"
leverage = 1.0