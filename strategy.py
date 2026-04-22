#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price action with weekly trend filter and volume confirmation.
# Uses 1-week high-low range to define trend context: long when price > weekly midpoint and above daily VWAP,
# short when price < weekly midpoint and below daily VWAP. Volume confirmation requires current volume > 2x
# 20-day average to filter low-quality signals. Designed for low-frequency, high-conviction trades
# (target: 10-25 trades/year) to minimize fee drag and work in both bull and bear markets via weekly trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend context (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly midpoint (average of weekly high and low)
    midpoint_1w = (high_1w + low_1w) / 2.0
    midpoint_1w_aligned = align_htf_to_ltf(prices, df_1w, midpoint_1w)
    
    # Calculate daily VWAP (volume-weighted average price)
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3.0
    vwap = (typical_price * prices['volume']).cumsum() / prices['volume'].cumsum()
    vwap = vwap.values
    
    # Calculate 20-day average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(midpoint_1w_aligned[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        midpoint = midpoint_1w_aligned[i]
        vwap_val = vwap[i]
        
        # Volume filter: current volume > 2 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: above weekly midpoint AND above VWAP + volume spike
            if price > midpoint and price > vwap_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: below weekly midpoint AND below VWAP + volume spike
            elif price < midpoint and price < vwap_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses VWAP in opposite direction
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below VWAP
                if price < vwap_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above VWAP
                if price > vwap_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyMidpoint_VWAP_Volume"
timeframe = "1d"
leverage = 1.0