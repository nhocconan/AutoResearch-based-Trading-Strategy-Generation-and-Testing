#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator system with daily momentum filter.
# Long when green lip (fast SMA) crosses above red jaw (slow SMA) with price above 1d EMA50.
# Short when red jaw crosses above green lip with price below 1d EMA50.
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lip=SMA(5,3).
# Uses 1d EMA50 as trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years).
name = "12h_WilliamsAlligator_1dEMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (Jaw=13, Teeth=8, Lip=5)
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    # Lip: 5-period SMA, shifted 3 bars forward
    lip = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Calculate EMA50 on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (wait for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need jaw data (13-period SMA + 8 shift)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lip[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        lip_val = lip[i]
        ema_trend = ema_50_aligned[i]
        
        if position == 0:
            # Enter long: green lip crosses above red jaw AND price above 1d EMA50
            if lip_val > jaw_val and lip[i-1] <= jaw[i-1] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: red jaw crosses above green lip AND price below 1d EMA50
            elif jaw_val > lip_val and jaw[i-1] <= lip[i-1] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when red jaw crosses above green lip
            if jaw_val > lip_val and jaw[i-1] <= lip[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when green lip crosses above red jaw
            if lip_val > jaw_val and lip[i-1] <= jaw[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals