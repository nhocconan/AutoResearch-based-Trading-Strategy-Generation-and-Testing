#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 14-day RSI with weekly pivot rejection + volume confirmation.
# RSI identifies overbought/oversold conditions (below 30 for long, above 70 for short).
# Weekly pivot levels act as dynamic support/resistance: only take longs when price > weekly pivot,
# shorts when price < weekly pivot to avoid counter-trend entries at major levels.
# Volume confirmation requires current volume > 1.3x 20-period average to filter weak breakouts.
# Designed for 6h timeframe to target 15-35 trades/year with strict confluence of signals.
# Works in bull markets via RSI oversold bounces above pivot, and in bear markets via
# RSI overbought rejections below pivot, avoiding false signals in sideways chop.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (standard formula: (H+L+C)/3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 14-period RSI on 6h data
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi[i]
        pivot_val = pivot_1w_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long conditions: oversold RSI + above weekly pivot + volume spike
            if rsi_val < 30 and price > pivot_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought RSI + below weekly pivot + volume spike
            elif rsi_val > 70 and price < pivot_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI returns to neutral zone or price crosses pivot
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to overbought or price breaks below pivot
                if rsi_val > 70 or price < pivot_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to oversold or price breaks above pivot
                if rsi_val < 30 or price > pivot_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_RSI_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0